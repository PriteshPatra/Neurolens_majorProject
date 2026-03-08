import sys
import threading

def send_notification(title: str, message: str):
    """
    Sends a native OS notification.
    Runs in a detached thread to prevent blocking the real-time YOLO inference loop.
    """
    def _notify():
        try:
            if sys.platform == "win32":
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(title, message, duration=5, threaded=True)
                except ImportError:
                    # Fallback to plyer
                    from plyer import notification
                    notification.notify(title=title, message=message, timeout=5)
            else:
                # Linux/macOS
                from plyer import notification
                notification.notify(title=title, message=message, timeout=5)
        except Exception as e:
            print(f"[OS Notifier] Failed to send notification: {e}")

    # Fire and forget
    threading.Thread(target=_notify, daemon=True).start()
