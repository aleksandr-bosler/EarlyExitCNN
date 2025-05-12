import os
import ee
import pandas as pd
from datetime import timedelta

PROJECT        = os.getenv('GOOGLE_CLOUD_PROJECT', 'my-project')
BUCKET         = os.getenv('GCS_BUCKET',        'my-bucket')
MAX_IMAGES     = 500
SCENES_FILE    = 'exported_scenes.txt'

if os.path.exists(SCENES_FILE):
    with open(SCENES_FILE, 'r') as f:
        used_ids = set(line.strip() for line in f if line.strip())
else:
    used_ids = set()

ee.Authenticate()
ee.Initialize(project=PROJECT)

SBANDS      = ['B2','B3','B4','B8','B11','B12']
CLOUD_MAX   = 70
PIXELS      = 224
SCALE       = 20
HALF_METERS = (PIXELS * SCALE) / 2

"""
# Borders of Australia
MIN_LAT, MAX_LAT = -44.0, -10.0
MIN_LON, MAX_LON = 112.0, 154.0
"""

# Borders of Amazonia
MIN_LAT, MAX_LAT = -20.0, 5.0
MIN_LON, MAX_LON = -75.0, -50.0


"""
# Borders of California
MIN_LAT, MAX_LAT = 32.0, 42.0
MIN_LON, MAX_LON = -124.5, -114.0
"""

csv_path = './fire_archive_SV-C2_607257_2021.csv'
with open(csv_path) as f:
    for skip, line in enumerate(f):
        if line.startswith('latitude,'):
            header_row = skip
            break

df = pd.read_csv(
    csv_path,
    skiprows=header_row,
    parse_dates=['acq_date'],
    dtype={'acq_time': str}
)
for c in ['type','frp','latitude','longitude']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['confidence','type','frp','latitude','longitude','acq_date','acq_time'])
df['confidence'] = df['confidence'].str.lower()
df = df[
    (df['confidence']=='h') &
    (df['type']==0) &
    (df['frp']>=50) &
    df['latitude'].between(MIN_LAT, MAX_LAT) &
    df['longitude'].between(MIN_LON, MAX_LON)
].reset_index(drop=True)

df['acq_time'] = df['acq_time'].str.zfill(4)
df['datetime'] = pd.to_datetime(
    df['acq_date'].dt.strftime('%Y-%m-%d') + df['acq_time'],
    format='%Y-%m-%d%H%M'
)

def start_export(i, row):
    dt = row['datetime']
    start_dt = dt - timedelta(hours=24)
    end_dt   = dt + timedelta(hours=24)
    lon, lat = row['longitude'], row['latitude']
    pt = ee.Geometry.Point([lon, lat])

    img = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(start_dt.isoformat(), end_dt.isoformat())
           .filterBounds(pt)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX))
           .select(SBANDS)
           .sort('CLOUDY_PIXEL_PERCENTAGE')
           .first()
    )
    if img is None:
        return False

    try:
        scene_id = img.get('system:index').getInfo()
    except Exception:
        return False
    if not scene_id or scene_id in used_ids:
        return False

    used_ids.add(scene_id)
    with open(SCENES_FILE, 'a') as f:
        f.write(scene_id + '\n')

    deg_per_meter = 1/111320.0
    d_deg = HALF_METERS * deg_per_meter
    region = [
        [lon-d_deg, lat-d_deg],
        [lon-d_deg, lat+d_deg],
        [lon+d_deg, lat+d_deg],
        [lon+d_deg, lat-d_deg],
    ]

    fname = f"fire_BRA_2021/pt{i:05d}_{row['acq_date'].date()}T{dt.time().strftime('%H%M')}_{lat:.4f}_{lon:.4f}"
    task = ee.batch.Export.image.toCloudStorage(
        image         = img,
        description   = f"fire_BRA_2021_{i:05d}",
        bucket        = BUCKET,
        fileNamePrefix= fname,
        scale         = SCALE,
        region        = region
    )
    task.start()

    return True

exported = 0
for i, row in df.iterrows():
    if exported >= MAX_IMAGES:
        break
    if start_export(i, row):
        exported += 1
        print(f"[{exported}/{MAX_IMAGES}] export started for row {i}")

print(f"{exported} unique exports launched (requested: {MAX_IMAGES}).")

