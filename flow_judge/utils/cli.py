import click

from .rubrics import load_rubric_templates, request_rubric


@click.group()
def cli():
    """Command line interface for Flow Judge."""
    pass


@cli.command(name="request-rubric")
@click.option("--title", prompt="Enter the title for the rubric request")
@click.option("--description", prompt="Enter a description for the rubric")
@click.option(
    "--similar-to",
    type=click.Choice(load_rubric_templates("rubrics").keys()),
    help="Specify a similar existing rubric",
)
def request_rubric_command(title, description, similar_to):
    """Create a new rubric request on GitHub."""
    request_rubric(title, description, similar_to)
    click.echo("Rubric request created successfully.")


if __name__ == "__main__":
    cli()
