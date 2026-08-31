"""Loading gmsh, and saying something useful when it will not load.

gmsh is the mesh generator behind every domain in this package, and it is
imported lazily at each call site so that a headless install which only reads
tables never pays for it. The cost of that laziness is where the failure
surfaces: three frames into ``inviscid_domain``, as an ``OSError`` naming a
shared library nobody asked for.

The failure is worth translating because it is not a missing package. The
``gmsh`` wheel on PyPI bundles a compiled library linked against X11 -- libXft,
libXcursor, libXinerama -- and pip cannot install system libraries, so
``pip install gmsh`` succeeds and ``import gmsh`` then fails on a machine
without them. Nothing about ``libXft.so.2: cannot open shared object file``
suggests that the fix is to install X11 development libraries into an
environment that is running headless and will never open a window.
"""

from __future__ import annotations

from typing import Any

__all__ = ["require_gmsh", "start_gmsh"]

#: The X11 libraries gmsh's bundled build links against, with the conda-forge
#: and Debian package that provides each.
_X11 = (
    ("libXft.so.2", "xorg-libxft", "libxft2"),
    ("libXcursor.so.1", "xorg-libxcursor", "libxcursor1"),
    ("libXinerama.so.1", "xorg-libxinerama", "libxinerama1"),
)


def require_gmsh() -> Any:
    """Import gmsh, or raise an error that says what to do about it."""
    try:
        import gmsh
    except ImportError as error:
        raise ImportError(
            "gmsh is not installed. It is an optional dependency of this package: "
            "`pip install gmsh`, or `conda install -c conda-forge gmsh`, which also "
            "brings the shared libraries the wheel does not."
        ) from error
    except OSError as error:
        conda = " ".join(package for _, package, _ in _X11)
        debian = " ".join(package for _, _, package in _X11)
        raise OSError(
            f"gmsh is installed but its compiled library will not load: {error}\n"
            "\n"
            "This is not a missing Python package. The gmsh wheel on PyPI bundles a "
            "library linked against X11, and pip cannot install system libraries, so "
            "`pip install gmsh` succeeds and the import then fails. The meshing itself "
            "is headless and never opens a window; the libraries are needed only "
            "because the same build also serves the GUI.\n"
            "\n"
            f"    conda install -c conda-forge {conda}\n"
            f"    apt-get install {debian}\n"
            "\n"
            "Installing gmsh itself from conda-forge pulls them in as dependencies, "
            "which is the more durable fix:\n"
            "\n"
            "    conda install -c conda-forge gmsh"
        ) from error
    return gmsh


def start_gmsh() -> Any:
    """Import and initialise gmsh, on whichever thread we happen to be on.

    ``gmsh.initialize`` installs a ``SIGINT`` handler so that Ctrl-C
    interrupts a long mesh, and Python only allows signal handlers to be
    installed from the main thread. Called from a worker it therefore raises
    ``ValueError: signal only works in main thread of the main interpreter``
    -- which is an obscure way to be told that meshing in the background is
    not allowed, and it is *not* what it means: the meshing is fine on a
    worker, only the convenience of Ctrl-C is not.

    So the handler is requested only where it can be installed. Meshing on a
    worker thread is what lets a notebook keep redrawing while gmsh runs, and
    it is worth the loss of Ctrl-C there, where the interrupt would not have
    reached the C call anyway.
    """
    import threading

    gmsh = require_gmsh()
    on_main = threading.current_thread() is threading.main_thread()
    gmsh.initialize(interruptible=on_main)
    return gmsh
