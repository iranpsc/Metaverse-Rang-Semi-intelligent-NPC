COMPOSE = docker compose --env-file .env

.PHONY: up up-with-vc down ps logs health migrate test smoke config production-up

up:
	$(COMPOSE) up -d --build

up-with-vc:
	$(COMPOSE) --profile voice-conversion up -d --build

production-up:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml --profile production --profile voice-conversion up -d --build

down:
	$(COMPOSE) --profile production --profile voice-conversion --profile training down

ps:
	$(COMPOSE) --profile production --profile voice-conversion ps

logs:
	$(COMPOSE) --profile voice-conversion logs -f --tail=200

health:
	python scripts/healthcheck.py

migrate:
	$(COMPOSE) exec web python manage.py migrate

test:
	$(COMPOSE) run --rm web python manage.py test AgentAPI tests --settings=tests.django_settings -v 2

smoke:
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR):/workspace:ro" agent-worker python /workspace/scripts/livekit_smoke.py --api-url http://web:8000 --livekit-url ws://livekit:7880 --wav-file "/workspace/$(WAV)"

config:
	$(COMPOSE) --profile production --profile voice-conversion config --quiet
