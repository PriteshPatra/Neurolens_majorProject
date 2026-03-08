import os
import sys

def set_brightness(target_brightness: int):
    """
    Sets the OS-level screen brightness across different platforms.
    target_brightness should be an integer between 0 and 100.
    """
    target_brightness = max(0, min(100, target_brightness))
    
    # Check if running in Docker without privileges
    if os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False):
        print(f"[OS Controller] Simulating brightness change to {target_brightness}% (Running in Docker)")
        return

    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(target_brightness)
    except Exception as e:
        print(f"[OS Controller] Native SBC failed: {e}. Attempting fallback...")
        if sys.platform.startswith("linux"):
            try:
                import subprocess
                # Get primary display
                output = subprocess.check_output(['xrandr']).decode('utf-8')
                primary = None
                for line in output.split('\n'):
                    if ' connected' in line:
                        primary = line.split()[0]
                        break
                
                if primary:
                    # xrandr brightness is 0.0 to 1.0
                    xrandr_brightness = max(0.1, target_brightness / 100.0) 
                    subprocess.run(['xrandr', '--output', primary, '--brightness', str(xrandr_brightness)], check=True)
            except Exception as fallback_e:
                print(f"[OS Controller] xrandr fallback failed: {fallback_e}")

