from typing import List, Union

import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from ..gui.from_np import QImage_from_np
from .QXWidget import QXWidget


class QXFixedLayeredImages(QXWidget):
    """
    A widget to show multiple stacked images in fixed area

    all images must have the same aspect ratio
    """
    def __init__(self, fwidth, height):
        super().__init__()
        self._fwidth = fwidth
        self._height = height
        self._qp = QPainter()
        self._images : List = []
        self._large_window : '_QXLargeImagesWindow' = None
        self.setToolTip('Double-click to open in a large window')

    def clear_images(self):
        self._images : List = []
        self._update_all()

    def _update_all(self):
        self.update()
        if self._large_window is not None:
            self._large_window.update()

    def has_large_window(self) -> bool:
        return self._large_window is not None

    def get_images(self) -> List:
        return [ image for image, _ in self._images ]

    def mouseDoubleClickEvent(self, ev : QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.open_large_window()
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)

    def open_large_window(self):
        if self._large_window is None:
            self._large_window = _QXLargeImagesWindow(self, title=self._find_title())
            self._large_window.show()
        self._large_window.raise_()
        self._large_window.activateWindow()

    def _on_large_window_closed(self):
        self._large_window = None

    def _find_title(self) -> str:
        # use title of parent section (e.g. 'Merged frame') if any
        w = self.parentWidget()
        while w is not None:
            label_title = getattr(w, 'label_title', None)
            if label_title is not None:
                return label_title.text()
            w = w.parentWidget()
        return ''

    def add_image(self, image, name=None):
        """
         image  QImage
                QPixmap
                np.ndarray  of uint8 dtype
        """
        saved_ref = None

        if not isinstance(image, QImage) and not isinstance(image, QPixmap):
            if isinstance(image, np.ndarray):
                saved_ref = image
                image = QImage_from_np(image)
            else:
                raise ValueError(f'Unsupported type of image {image.__class__}')

        self._images.append( (image, saved_ref) )
        self._update_all()

    def sizeHint(self):
        return QSize(self._fwidth, self._height)

    def paintEvent(self, event):
        super().paintEvent(event)

        qp = self._qp
        qp.begin(self)
        qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self._fwidth
        h = self._height

        w_half = w /2
        h_half = h /2
        a = w/h

        for image, _ in self._images:

            size = image.size()
            ap = size.width() / size.height()

            if ap > a:
                ph_fit = h * (a / ap)
                rect = QRect(0, int(h_half-ph_fit/2), int(w), int(ph_fit) )
            elif ap < a:
                pw_fit = w * (ap / a)
                rect = QRect(int(w_half-pw_fit/2), 0, int(pw_fit), int(h) )
            else:
                rect = self.rect()

            if isinstance(image, QImage):
                qp.drawImage(rect, image, image.rect())
            elif isinstance(image, QPixmap):
                qp.drawPixmap(rect, image, image.rect())

        qp.end()


class _QXLargeImagesWindow(QWidget):
    """
    Top-level window showing images of QXFixedLayeredImages in large size.

    image smaller than the window : shown at real size, centered on black background
    image larger than the window  : scaled down keeping aspect ratio

    Double-click : toggle fullscreen
    Esc          : exit fullscreen / close
    """
    def __init__(self, owner : QXFixedLayeredImages, title : str = ''):
        super().__init__(owner.window(), Qt.WindowType.Window)
        self._owner = owner
        self._qp = QPainter()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setWindowTitle(title if title else 'Preview')
        self.setMinimumSize(160, 120)

        # initial size: image real size, limited to 90% of the screen
        w, h = 1280, 720
        images = owner.get_images()
        if len(images) != 0:
            size = images[0].size()
            w, h = size.width(), size.height()
        screen = owner.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(w, int(avail.width()*0.9))
            h = min(h, int(avail.height()*0.9))
        self.resize(max(w, 320), max(h, 240))

    def mouseDoubleClickEvent(self, ev : QMouseEvent):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        ev.accept()

    def keyPressEvent(self, ev : QKeyEvent):
        if ev.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
            ev.accept()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev : QCloseEvent):
        self._owner._on_large_window_closed()
        super().closeEvent(ev)

    def paintEvent(self, event):
        qp = self._qp
        qp.begin(self)
        qp.fillRect(self.rect(), Qt.GlobalColor.black)
        qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        W, H = self.width(), self.height()
        for image in self._owner.get_images():
            size = image.size()
            iw, ih = size.width(), size.height()
            if iw == 0 or ih == 0:
                continue
            # never upscale, downscale to fit keeping aspect ratio
            scale = min(1.0, W / iw, H / ih)
            dw, dh = int(iw*scale), int(ih*scale)
            rect = QRect( (W-dw)//2, (H-dh)//2, dw, dh)

            if isinstance(image, QImage):
                qp.drawImage(rect, image, image.rect())
            elif isinstance(image, QPixmap):
                qp.drawPixmap(rect, image, image.rect())

        qp.end()
