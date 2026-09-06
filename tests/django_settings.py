from MetaRangNPC.settings import *  # noqa: F403


ROOT_URLCONF = "tests.django_urls"
CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
