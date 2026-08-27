from typer.testing import CliRunner
from fraudshield.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "FraudShield ML CLI" in result.stdout

def test_cli_validate_missing_config():
    result = runner.invoke(app, ["validate-data"])
    assert result.exit_code != 0
    assert "Missing option" in result.stdout
