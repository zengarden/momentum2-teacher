format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r momentum_teacher
	black --line-length 120 .

style_check:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r momentum_teacher
	black --line-length 120 --diff --check .

unittest:
	pytest --cov=momentum_teacher --cov-report=html --cov-report term test
