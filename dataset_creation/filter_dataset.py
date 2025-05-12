import os, sys, shutil
import numpy as np
import cv2

FOLDER             = 'GCS/BRA_2021'

BAD_LIST_PATH      = os.path.join(FOLDER, 'bad_files.txt')
SMOKE_LIST_PATH    = os.path.join(FOLDER, 'smoke_only.txt')

MOVE_TO_BAD        = True
MOVE_TO_SMOKE      = True

BAD_FOLDER         = os.path.join(FOLDER, 'bad')
SMOKE_FOLDER       = os.path.join(FOLDER, 'smoke_only')

RGB_BANDS          = [2, 3, 5]  #[2, 3, 4]
SWIR_BAND          = 5           #10

SCALE_FACTOR       = 2.5
WINDOW_WIDTH       = 1200
WINDOW_HEIGHT      = 600

def view_and_mark_npz(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.npz'))
    total = len(files)

    bad   = set(open(BAD_LIST_PATH,   'r', encoding='utf-8').read().splitlines()) if os.path.exists(BAD_LIST_PATH)   else set()
    smoke = set(open(SMOKE_LIST_PATH, 'r', encoding='utf-8').read().splitlines()) if os.path.exists(SMOKE_LIST_PATH) else set()

    if MOVE_TO_BAD   and not os.path.isdir(BAD_FOLDER):   os.makedirs(BAD_FOLDER)
    if MOVE_TO_SMOKE and not os.path.isdir(SMOKE_FOLDER): os.makedirs(SMOKE_FOLDER)

    cv2.namedWindow('Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Viewer', WINDOW_WIDTH, WINDOW_HEIGHT)

    for idx, fname in enumerate(files, 1):
        path = os.path.join(folder, fname)
        print(f"\n[{idx}/{total}] {fname}")

        try:
            data = np.load(path)
            key  = 'image' if 'image' in data else list(data.keys())[0]
            img  = data[key]
            data.close()
        except Exception as e:
            continue

        if img.ndim != 3:
            continue

        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            img = ((img - mn)/(mx - mn)*255).astype(np.uint8) if mx>mn else np.zeros_like(img, np.uint8)

        try:
            rgb = np.stack([img[:,:,b] for b in RGB_BANDS], axis=-1)
        except:
            rgb = img[:,:,:3]

        sw   = img[:,:,SWIR_BAND] if SWIR_BAND < img.shape[2] else img[:,:,0]
        swir = cv2.applyColorMap(sw, cv2.COLORMAP_HOT)

        combined = np.hstack((rgb, swir))
        nh, nw = int(combined.shape[0]*SCALE_FACTOR), int(combined.shape[1]*SCALE_FACTOR)
        combined = cv2.resize(combined, (nw, nh), interpolation=cv2.INTER_LINEAR)

        status = 'BAD' if fname in bad else ('SMOKE ONLY' if fname in smoke else 'OK')
        title  = f"[{idx}/{total}] {fname} [{status}]"
        cv2.setWindowTitle('Viewer', title)
        cv2.imshow('Viewer', combined)

        for _ in range(3):
            cv2.waitKey(1)

        key = cv2.waitKey(0)

        if   key == 27:  # Esc
            break
        elif key in (13,10):  # Enter
            continue
        elif key in (ord('m'), ord('M')):  # mark bad
            if fname not in bad:
                bad.add(fname)
                with open(BAD_LIST_PATH, 'a', encoding='utf-8') as bf:
                    bf.write(fname + "\n")
            if MOVE_TO_BAD:
                shutil.move(path, os.path.join(BAD_FOLDER, fname))
        elif key in (ord('s'), ord('S')):  # mark smoke-only
            if fname not in smoke:
                smoke.add(fname)
                with open(SMOKE_LIST_PATH, 'a', encoding='utf-8') as sf:
                    sf.write(fname + "\n")
            if MOVE_TO_SMOKE:
                shutil.move(path, os.path.join(SMOKE_FOLDER, fname))
        elif key in (ord('d'), ord('D')):  # delete
            os.remove(path)
        else:
            print("  (Enter=keep, m/M=bad, s/S=smoke_only, d/D=delete, Esc=quit)")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_and_mark_npz(FOLDER)
