"""Touch ID user-presence gate for key release (macOS only).

Touch ID never exposes fingerprint images; it only answers "the enrolled Mac user touched the
sensor". SecureFinger uses it as the platform biometric authorization step before the
SecureContext signs an ACCEPT decision (the role BiometricPrompt plays on Android).

Requires:  pip install pyobjc-framework-LocalAuthentication
Self-test: python -m securefinger.touchid
"""
import sys
import threading


def available() -> tuple[bool, str]:
    """(ok, reason). ok is True only on a Mac with Touch ID enrolled and PyObjC installed."""
    if sys.platform != "darwin":
        return False, "Touch ID needs macOS"
    try:
        from LocalAuthentication import LAContext, LAPolicyDeviceOwnerAuthenticationWithBiometrics
    except ImportError:
        return False, "install pyobjc-framework-LocalAuthentication"
    ok, err = LAContext.new().canEvaluatePolicy_error_(LAPolicyDeviceOwnerAuthenticationWithBiometrics, None)
    if not ok:
        return False, f"Touch ID unavailable: {err.localizedDescription() if err else 'not enrolled'}"
    return True, "ready"


def authenticate(reason: str, timeout: float = 60.0) -> bool:
    """Show the system Touch ID prompt. True only if the user's fingerprint matched."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    from LocalAuthentication import LAContext, LAPolicyDeviceOwnerAuthenticationWithBiometrics

    ctx = LAContext.new()
    done, result = threading.Event(), {"ok": False}

    def reply(success, error):
        result["ok"] = bool(success)
        result["error"] = error
        done.set()

    ctx.evaluatePolicy_localizedReason_reply_(LAPolicyDeviceOwnerAuthenticationWithBiometrics, reason, reply)
    if not done.wait(timeout):
        ctx.invalidate()
        return False
    return result["ok"]


if __name__ == "__main__":
    ok, why = available()
    print("Touch ID available:", ok, "-", why)
    if ok:
        print("Touch the sensor...")
        print("Authenticated:", authenticate("SecureFinger Touch ID self-test"))
