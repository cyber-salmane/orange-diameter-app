from datetime import datetime, timedelta

from utils import validate_email, validate_username, validate_password, is_token_expired


def test_validate_email():
    assert validate_email("user@example.com")
    assert not validate_email("user@example")
    assert not validate_email("@example.com")


def test_validate_username():
    assert validate_username("valid_user")
    assert validate_username("user-123")
    assert not validate_username("no spaces")
    assert not validate_username("a")


def test_validate_password():
    valid, msg = validate_password("StrongPass1")
    assert valid
    assert msg == ""

    valid, msg = validate_password("weak")
    assert not valid
    assert "8 caractères" in msg


def test_is_token_expired():
    recent = (datetime.now() - timedelta(minutes=30)).isoformat()
    expired = (datetime.now() - timedelta(hours=2)).isoformat()

    assert not is_token_expired(recent)
    assert is_token_expired(expired)
