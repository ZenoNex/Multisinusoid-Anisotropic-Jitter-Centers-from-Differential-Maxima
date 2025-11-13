#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multisinusoid Anisotropic Jitter – Centres from Differential Maxima
-----------------------------------------------------------------
* Click on bias map → add centre (snaps to local max)
* Edit table → live update
* Animated waves (phase offset over time)
* Safe OpenCV + Qt interop
"""

import sys
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QPushButton, QVBoxLayout, QLabel
import time


# ----------------------------------------------------------------------
# 1. Anisotropy field (gradient direction + eccentricity)
# ----------------------------------------------------------------------
def compute_anisotropy_field(bias: np.ndarray):
    gy, gx = np.gradient(bias)
    mag = np.hypot(gx, gy) + 1e-9
    theta = np.arctan2(gy, gx)
    ecc = 1.0 - np.exp(-3.0 * mag / mag.mean())
    return theta, ecc, mag


# ----------------------------------------------------------------------
# 2. Find local maxima (centres)
# ----------------------------------------------------------------------
def find_local_maxima(bias: np.ndarray, min_distance=40, rel_threshold=0.25):
    dilated = cv2.dilate(bias, np.ones((3, 3), np.uint8))
    maxima = (dilated == bias)
    thresh = bias.max() * rel_threshold
    mask = (bias > thresh) & maxima
    ys, xs = np.where(mask)
    points = sorted([(int(y), int(x), float(bias[y, x])) for y, x in zip(ys, xs)],
                    key=lambda p: -p[2])

    kept = []
    for p in points:
        if all((p[0] - k[0])**2 + (p[1] - k[1])**2 > min_distance**2 for k in kept):
            kept.append(p)
    return kept


# ----------------------------------------------------------------------
# 3. Core filter – sinusoids from centres
# ----------------------------------------------------------------------
class CenterJitterFilter:
    def __init__(self, bias_img: np.ndarray):
        self.bias = bias_img.astype(np.float32) / 255.0
        self.H, self.W = self.bias.shape
        self.theta, self.ecc, self.gradmag = compute_anisotropy_field(self.bias)
        self.centers = []

    def set_centers(self, centers):
        self.centers = centers

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)

        Dx = np.zeros((H, W), np.float32)
        Dy = np.zeros((H, W), np.float32)

        for c in self.centers:
            cx, cy = c['cx'], c['cy']
            w, f, a, ph, ori = c['weight'], c['freq'], c['amp'], c['phase'], c['orient']

            vx = xg - cx
            vy = yg - cy
            proj = vx * np.cos(ori) + vy * np.sin(ori)
            phase_field = 2 * np.pi * f * proj + ph
            wave = w * a * np.sin(phase_field)

            Dx += wave * np.cos(ori)
            Dy += wave * np.sin(ori)

        # Anisotropic projection
        major_x = np.cos(self.theta)
        major_y = np.sin(self.theta)
        minor_x = -major_y
        minor_y = major_x

        D_par = Dx * major_x + Dy * major_y
        D_perp = Dx * minor_x + Dy * minor_y

        scale_par = 1.0 + self.ecc
        scale_perp = 1.0 - self.ecc

        Dx_aniso = D_par * scale_par * major_x + D_perp * scale_perp * minor_x
        Dy_aniso = D_par * scale_par * major_y + D_perp * scale_perp * minor_y

        map_x = (xg + Dx_aniso).astype(np.float32)
        map_y = (yg + Dy_aniso).astype(np.float32)

        out = cv2.remap(src_bgr, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        return out


# ----------------------------------------------------------------------
# 4. Safe NumPy → QPixmap
# ----------------------------------------------------------------------
def numpy_to_qpixmap(img_bgr: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


# ----------------------------------------------------------------------
# 5. GUI Components
# ----------------------------------------------------------------------
class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, int)  # y, x

    def mousePressEvent(self, ev):
        if not self.pixmap():
            return
        pw, ph = self.pixmap().width(), self.pixmap().height()
        lw, lh = self.width(), self.height()
        x = int(ev.pos().x() * pw / lw)
        y = int(ev.pos().y() * ph / lh)
        self.clicked.emit(y, x)


class CenterTable(QtWidgets.QTableWidget):
    changed = QtCore.pyqtSignal()  # Custom signal (was dataChanged)

    def __init__(self):
        super().__init__(0, 8)
        headers = ["#", "Y", "X", "Weight", "Freq", "Amp", "Phase", "Orient°"]
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

    def update_from_centers(self, centers):
        self.blockSignals(True)
        self.setRowCount(0)
        for i, c in enumerate(centers):
            self.insertRow(i)
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            fields = ["cy", "cx", "weight", "freq", "amp", "phase", "orient"]
            for col, key in enumerate(fields, 1):
                val = c[key]
                if key == "orient":
                    val = np.rad2deg(val)
                item = QtWidgets.QTableWidgetItem(f"{val:.4f}")
                item.setData(QtCore.Qt.UserRole, c)
                self.setItem(i, col, item)
        self.blockSignals(False)
        self.changed.emit()  # Emit custom signal

    def get_centers(self):
        centers = []
        for row in range(self.rowCount()):
            base = self.item(row, 1).data(QtCore.Qt.UserRole).copy()
            base.update({
                'cy': float(self.item(row, 1).text()),
                'cx': float(self.item(row, 2).text()),
                'weight': float(self.item(row, 3).text()),
                'freq': float(self.item(row, 4).text()),
                'amp': float(self.item(row, 5).text()),
                'phase': float(self.item(row, 6).text()),
                'orient': np.deg2rad(float(self.item(row, 7).text())),
            })
            centers.append(base)
        return centers


# ----------------------------------------------------------------------
# 6. Main Window
# ----------------------------------------------------------------------
class JitterWorker(QThread):
    finished = pyqtSignal(np.ndarray)      # emits the jittered BGR image
    def __init__(self, filter_obj, src_bgr, centers):
        super().__init__()
        self.filter = filter_obj
        self.src = src_bgr.copy()
        self.centers = centers
        self._abort = False

    def run(self):
        if self._abort: return
        # ---- apply animation offset (same as before) ----
        for c in self.centers:
            c['phase'] = c.get('base_phase', 0.0) + getattr(self.filter, 'phase_offset', 0.0)
        self.filter.set_centers(self.centers)
        out = self.filter.apply(self.src)
        if not self._abort:
            self.finished.emit(out)

    def stop(self):
        self._abort = True


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bias_path, src_path):
        super().__init__()
        self.setWindowTitle("Anisotropic Jitter – Fast + Save")
        self.resize(1450, 780)

        # ---------- load images ----------
        self.bias_gray = cv2.imread(bias_path, cv2.IMREAD_GRAYSCALE)
        self.src_bgr   = cv2.imread(src_path)
        if self.bias_gray is None or self.src_bgr is None:
            raise FileNotFoundError("Failed to load images")
        if self.src_bgr.shape[:2] != self.bias_gray.shape[:2]:
            self.src_bgr = cv2.resize(self.src_bgr,
                                      (self.bias_gray.shape[1], self.bias_gray.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

        self.filter = CenterJitterFilter(self.bias_gray)
        self.phase_offset = 0.0

        # ---------- UI ----------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # left – bias map
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self.bias_label = ClickableLabel()
        self.bias_label.setMinimumSize(420, 420)
        self.bias_label.setStyleSheet("background:#111; border:1px solid #444;")
        left_layout.addWidget(self.bias_label)

        btn_auto = QtWidgets.QPushButton("Auto-Detect Centres")
        btn_auto.clicked.connect(self.auto_detect)
        left_layout.addWidget(btn_auto)
        layout.addWidget(left, stretch=1)

        # right – result + table + buttons
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        self.result_label = QtWidgets.QLabel()
        self.result_label.setMinimumSize(600, 500)
        self.result_label.setStyleSheet("background:#222; border:1px solid #444;")
        right_layout.addWidget(self.result_label)

        self.table = CenterTable()
        self.table.itemChanged.connect(lambda item: self.schedule_update())
        right_layout.addWidget(self.table)

        btn_del = QtWidgets.QPushButton("Delete Selected")
        btn_del.clicked.connect(self.delete_selected)
        right_layout.addWidget(btn_del)

        btn_save = QtWidgets.QPushButton("Save Current Frame")
        btn_save.clicked.connect(self.save_current_frame)
        right_layout.addWidget(btn_save)

        # ← ADD RELOAD BUTTON HERE (right after Save)
        btn_reload = QtWidgets.QPushButton("Reload Images")
        btn_reload.clicked.connect(self.reload_images)
        right_layout.addWidget(btn_reload)


        layout.addWidget(right, stretch=2)

        # ---------- connections ----------
        self.bias_label.clicked.connect(self.add_centre_at_click)
        self.table.changed.connect(self.schedule_update)

        # ---------- background worker ----------
        self.worker = None
        self.last_centers_hash = None

        # ---------- timer (animation only) ----------
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self.advance_animation)
        self.anim_timer.start(80)   # ~12 FPS animation

        # ---------- init ----------
        self.auto_detect()
        self.refresh_bias_display()

    # ------------------------------------------------------------------
    def schedule_update(self):
        """Called whenever table or centres change – throttles heavy work."""
        centers = self.table.get_centers()
        h = hash(tuple((c['cx'], c['cy'], c['weight'], c['freq'], c['amp'],
                        c['phase'], c['orient']) for c in centers))
        if h == self.last_centers_hash:
            return                     # no change → skip
        self.last_centers_hash = h

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.worker = JitterWorker(self.filter, self.src_bgr, centers)
        self.worker.finished.connect(self.display_result)
        self.worker.start()

    # ------------------------------------------------------------------
    def advance_animation(self):
        """Only updates phase – very cheap."""
        self.phase_offset += 0.08
        # copy current phase into filter for worker
        self.filter.phase_offset = self.phase_offset
        # trigger a recompute only if a worker isn’t already running
        if not (self.worker and self.worker.isRunning()):
            self.schedule_update()

    # ------------------------------------------------------------------
    def display_result(self, out_bgr):
        pixmap = numpy_to_qpixmap(out_bgr).scaled(
            self.result_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.result_label.setPixmap(pixmap)
        self.refresh_bias_display()

    # ------------------------------------------------------------------
    def save_current_frame(self):
        """Save the *current* displayed image (not a new computation)."""
        pix = self.result_label.pixmap()
        if pix:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Frame", "output.png", "PNG Images (*.png)")
            if path:
                pix.save(path, "PNG")

    # ← PASTE reload_images RIGHT HERE
    def reload_images(self):
        bias_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Reload Bias Image", "", "Images (*.png *.jpg *.bmp)")
        src_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Reload Source Image", "", "Images (*.png *.jpg *.bmp)")
        if bias_path and src_path:
            self.bias_gray = cv2.imread(bias_path, cv2.IMREAD_GRAYSCALE)
            self.src_bgr = cv2.imread(src_path)
            if self.src_bgr.shape[:2] != self.bias_gray.shape[:2]:
                self.src_bgr = cv2.resize(
                    self.src_bgr,
                    (self.bias_gray.shape[1], self.bias_gray.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
            self.filter = CenterJitterFilter(self.bias_gray)
            self.auto_detect()

    # ------------------------------------------------------------------
    def refresh_bias_display(self):
        viz = cv2.cvtColor(self.bias_gray, cv2.COLOR_GRAY2BGR)
        for c in self.filter.centers:
            pt = (int(c['cx']), int(c['cy']))
            cv2.circle(viz, pt, 10, (0, 255, 0), 2)
            cv2.circle(viz, pt, 3, (0, 0, 255), -1)
        self.bias_label.setPixmap(numpy_to_qpixmap(viz).scaled(
            self.bias_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    # ------------------------------------------------------------------
    def auto_detect(self):
        maxima = find_local_maxima(self.bias_gray, min_distance=40, rel_threshold=0.25)
        centers = [
            dict(cx=x, cy=y, weight=v / 255.0, freq=0.02, amp=12.0, phase=0.0,
                 orient=0.0, base_phase=0.0)
            for y, x, v in maxima
        ]
        self.filter.set_centers(centers)
        self.table.update_from_centers(centers)
        self.refresh_bias_display()

    # ------------------------------------------------------------------
    def add_centre_at_click(self, y, x):
        h, w = self.bias_gray.shape
        y0, y1 = max(0, y - 15), min(h, y + 16)
        x0, x1 = max(0, x - 15), min(w, x + 16)
        patch = self.bias_gray[y0:y1, x0:x1]
        if patch.size == 0:
            return
        ly, lx = np.unravel_index(patch.argmax(), patch.shape)
        cy, cx = y0 + ly, x0 + lx
        val = self.bias_gray[cy, cx] / 255.0

        new_c = dict(cx=cx, cy=cy, weight=val, freq=0.02, amp=12.0,
                     phase=0.0, orient=0.0, base_phase=0.0)
        self.filter.centers.append(new_c)
        self.table.update_from_centers(self.filter.centers)
        self.refresh_bias_display()

    # ------------------------------------------------------------------
    def delete_selected(self):
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        self.filter.centers = [c for i, c in enumerate(self.filter.centers)
                               if i not in rows]
        self.table.update_from_centers(self.filter.centers)
        self.refresh_bias_display()

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Create a tiny starter window to pick files
    dialog = QtWidgets.QWidget()
    dialog.setWindowTitle("Select Images")
    dialog.resize(400, 150)
    layout = QVBoxLayout(dialog)

    lbl = QLabel("Select Bias Image (grayscale) and Source Image")
    layout.addWidget(lbl)

    btn_bias = QPushButton("Choose Bias Image")
    btn_src  = QPushButton("Choose Source Image")
    btn_go   = QPushButton("Start Jitter")

    layout.addWidget(btn_bias)
    layout.addWidget(btn_src)
    layout.addWidget(btn_go)

    bias_path = [None]
    src_path  = [None]

    def pick_bias():
        path, _ = QFileDialog.getOpenFileName(dialog, "Select Bias Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            bias_path[0] = path
            btn_bias.setText(f"Bias: {path.split('/')[-1]}")

    def pick_src():
        path, _ = QFileDialog.getOpenFileName(dialog, "Select Source Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            src_path[0] = path
            btn_src.setText(f"Source: {path.split('/')[-1]}")

    def start():
        if bias_path[0] and src_path[0]:
            dialog.close()
            win = MainWindow(bias_path[0], src_path[0])
            win.show()
        else:
            QtWidgets.QMessageBox.warning(dialog, "Missing", "Please select both images.")

    btn_bias.clicked.connect(pick_bias)
    btn_src.clicked.connect(pick_src)
    btn_go.clicked.connect(start)

    dialog.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
