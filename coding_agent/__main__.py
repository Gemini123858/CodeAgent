if __package__:
    from .cli import main
else:
    # Support direct execution with ``python coding_agent/__main__.py`` or
    # ``cd coding_agent && python __main__.py``. In that mode Python does not
    # add the package's parent directory to sys.path automatically.
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from coding_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
