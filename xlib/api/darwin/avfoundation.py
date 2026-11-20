from typing import List


def get_video_input_devices_names() -> List[str]:
    """
    returns a list of camera names, in the same order as OpenCV indexes them
    on macOS (cv2.CAP_AVFOUNDATION / cv2.CAP_ANY).

    OpenCV enumerates AVCaptureDevice of media types Video and Muxed,
    then sorts them by uniqueID. We replicate it so that list index == cv2 index.

    returns empty list if pyobjc AVFoundation is not available.
    """
    try:
        import AVFoundation as AVF
    except ImportError:
        return []

    try:
        devices = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)) + \
                  list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeMuxed))
        devices.sort(key=lambda d: str(d.uniqueID()))
        return [ str(d.localizedName()) for d in devices ]
    except Exception as e:
        print(f'AVFoundation devices enumeration error: {e}')
        return []
