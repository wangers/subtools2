# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import gzip
import io
import os
import posixpath
import re
import stat
from glob import has_magic
from pathlib import Path


# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/workarounds.py#AltGzipFilePatched
def gzip_open_patch(
    filename,
    mode="rb",
    compresslevel=9,  # compat with Py 3.6
    encoding=None,
    errors=None,
    newline=None,
):
    """
    Open a gzip-compressed file in binary or text mode. To handle "trailing garbage" in gzip files.
    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    gz_mode = mode.replace("t", "")
    if isinstance(filename, (str, bytes, os.PathLike)):
        binary_file = AltGzipFile(filename, gz_mode, compresslevel)
    elif hasattr(filename, "read") or hasattr(filename, "write"):
        binary_file = AltGzipFile(None, gz_mode, compresslevel, filename)
    else:
        raise TypeError("filename must be a str or bytes object, or a file")

    if "t" in mode:
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


class AltGzipFile(gzip.GzipFile):
    """
    This is a workaround for Python's stdlib gzip module
    not implementing gzip decompression correctly...
    Command-line gzip is able to discard "trailing garbage" in gzipped files,
    but Python's gzip is not.
    """

    def read(self, size=-1):
        chunks = []
        try:
            if size < 0:
                while True:
                    chunk = self.read1()
                    if not chunk:
                        break
                    chunks.append(chunk)
            else:
                while size > 0:
                    chunk = self.read1(size)
                    if not chunk:
                        break
                    size -= len(chunk)
                    chunks.append(chunk)
        except OSError as e:
            if not chunks or not str(e).startswith("Not a gzipped file"):
                raise

        return b"".join(chunks)


class FsspecLocalGlob:
    """
    A glob from `fsspec <http://github.com/fsspec/filesystem_spec>`_.

    Here are some behaviors specific to fsspec glob that are different from
    glob.glob, Path.glob, Path.match or fnmatch:

    - ``'*'`` matches only first level items
    - ``'**'`` matches all items
    - ``'**/*'`` matches all at least second level items

        For example, ``glob.glob('**/*', recursive=True)``, the last ``/*`` is invalid as greedy mode of first pattern ``'**'``.

    """

    root_marker = "/"
    protocol = "file"

    @classmethod
    def info(cls, path, **kwargs):
        if isinstance(path, os.DirEntry):
            # scandir DirEntry
            out = path.stat(follow_symlinks=False)
            link = path.is_symlink()
            if path.is_dir(follow_symlinks=False):
                t = "directory"
            elif path.is_file(follow_symlinks=False):
                t = "file"
            else:
                t = "other"
            path = cls._strip_protocol(path.path)
        else:
            # str or path-like
            path = cls._strip_protocol(path)
            out = os.stat(path, follow_symlinks=False)
            link = stat.S_ISLNK(out.st_mode)
            if link:
                out = os.stat(path, follow_symlinks=True)
            if stat.S_ISDIR(out.st_mode):
                t = "directory"
            elif stat.S_ISREG(out.st_mode):
                t = "file"
            else:
                t = "other"
        result = {
            "name": path,
            "size": out.st_size,
            "type": t,
            "created": out.st_ctime,
            "islink": link,
        }
        for field in ["mode", "uid", "gid", "mtime", "ino", "nlink"]:
            result[field] = getattr(out, "st_" + field)
        if result["islink"]:
            result["destination"] = os.readlink(path)
            try:
                out2 = os.stat(path, follow_symlinks=True)
                result["size"] = out2.st_size
            except IOError:
                result["size"] = 0
        return result

    @classmethod
    def glob(cls, path, **kwargs):
        """
        Find files by glob-matching.

        Here are some behaviors specific to fsspec glob that are different from glob.glob, Path.glob, Path.match or fnmatch:

        - ``'*'`` matches only first level items
        - ``'**'`` matches all items
        - ``'**/*'`` matches all at least second level items

            e.g., glob.glob('**/*', recursive=True), the last '/*' is invalid as greedy mode of first pattern '**'.

        If the path ends with '/' and does not contain `"*"`, it is essentially
        the same as ``ls(path)``, returning only files.

        We support ``"**"``,
        ``"?"`` and ``"[..]"``. We do not support '^' for pattern negation.

        Search path names that contain embedded characters special to this
        implementation of glob may not produce expected results;
        e.g., `'foo/bar/*starredfilename*'`.

        kwargs are passed to ``ls``.
        """

        ends = path.endswith("/")
        path = cls._strip_protocol(path)
        indstar = path.find("*") if path.find("*") >= 0 else len(path)
        indques = path.find("?") if path.find("?") >= 0 else len(path)
        indbrace = path.find("[") if path.find("[") >= 0 else len(path)

        ind = min(indstar, indques, indbrace)

        detail = kwargs.pop("detail", False)

        if not has_magic(path):
            root = path
            depth = 1
            if ends:
                path += "/*"
            elif cls.exists(path):
                if not detail:
                    return [path]
                else:
                    return {path: cls.info(path)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:ind]:
            ind2 = path[:ind].rindex("/")
            root = path[: ind2 + 1]
            depth = None if "**" in path else path[ind2 + 1 :].count("/") + 1
        else:
            root = ""
            depth = None if "**" in path else path[ind + 1 :].count("/") + 1

        allpaths = cls.find(root, maxdepth=depth, withdirs=True, detail=True, **kwargs)
        # Escape characters special to python regex, leaving our supported
        # special characters in place.
        # See https://www.gnu.org/software/bash/manual/html_node/Pattern-Matching.html
        # for shell globbing details.
        pattern = (
            "^"
            + (
                path.replace("\\", r"\\")
                .replace(".", r"\.")
                .replace("+", r"\+")
                .replace("//", "/")
                .replace("(", r"\(")
                .replace(")", r"\)")
                .replace("|", r"\|")
                .replace("^", r"\^")
                .replace("$", r"\$")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .rstrip("/")
                .replace("?", ".")
            )
            + "$"
        )
        pattern = re.sub("[*]{2}", "=PLACEHOLDER=", pattern)
        pattern = re.sub("[*]", "[^/]*", pattern)
        pattern = re.compile(pattern.replace("=PLACEHOLDER=", ".*"))
        out = {
            p: allpaths[p]
            for p in sorted(allpaths)
            if pattern.match(p.replace("//", "/").rstrip("/"))
        }
        if detail:
            return out
        else:
            return list(out)

    @classmethod
    def isfile(cls, path):
        """Is this entry file-like?"""
        try:
            return cls.info(path)["type"] == "file"
        except:  # noqa: E722
            return False

    @classmethod
    def exists(cls, path, **kwargs):
        """Is there a file at the given path"""
        try:
            cls.info(path, **kwargs)
            return True
        except:  # noqa: E722
            # any exception allowed bar FileNotFoundError?
            return False

    @classmethod
    def walk(cls, path, maxdepth=None, topdown=True, **kwargs):
        """Return all files belows path

        List all files, recursing into subdirectories; output is iterator-style,
        like ``os.walk()``. For a simple list of files, ``find()`` is available.

        When topdown is True, the caller can modify the dirnames list in-place (perhaps
        using del or slice assignment), and walk() will
        only recurse into the subdirectories whose names remain in dirnames;
        this can be used to prune the search, impose a specific order of visiting,
        or even to inform walk() about directories the caller creates or renames before
        it resumes walk() again.
        Modifying dirnames when topdown is False has no effect. (see os.walk)

        Note that the "files" outputted will include anything that is not
        a directory, such as links.

        Parameters
        ----------
        path: str
            Root to recurse into
        maxdepth: int
            Maximum recursion depth. None means limitless, but not recommended
            on link-based file-systems.
        topdown: bool (True)
            Whether to walk the directory tree from the top downwards or from
            the bottom upwards.
        **kwargs:
            passed to ``ls``.
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        path = cls._strip_protocol(path)
        full_dirs = {}
        dirs = {}
        files = {}

        detail = kwargs.pop("detail", False)
        try:
            listing = cls.ls(path, detail=True, **kwargs)
        except (FileNotFoundError, IOError):
            if detail:
                return path, {}, {}
            return path, [], []

        for info in listing:
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            pathname = info["name"].rstrip("/")
            name = pathname.rsplit("/", 1)[-1]
            if info["type"] == "directory" and pathname != path:
                # do not include "self" path
                full_dirs[name] = pathname
                dirs[name] = info
            elif pathname == path:
                # file-like with same name as give path
                files[""] = info
            else:
                files[name] = info

        if not detail:
            dirs = list(dirs)
            files = list(files)

        if topdown:
            # Yield before recursion if walking top down
            yield path, dirs, files

        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                if not topdown:
                    yield path, dirs, files
                return

        for d in dirs:
            yield from cls.walk(
                full_dirs[d],
                maxdepth=maxdepth,
                detail=detail,
                topdown=topdown,
                **kwargs,
            )

        if not topdown:
            # Yield after recursion if walking bottom up
            yield path, dirs, files

    @classmethod
    def find(cls, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        """List all files below path.

        Like posix ``find`` command without conditions

        Parameters
        ----------
        path : str
        maxdepth: int or None
            If not None, the maximum number of levels to descend
        withdirs: bool
            Whether to include directory paths in the output. This is True
            when used by glob, but users usually only want files.
        **kwargs:
            passed to ``ls``.
        """
        # TODO: allow equivalent of -name parameter
        path = cls._strip_protocol(path)
        out = dict()
        for _, dirs, files in cls.walk(path, maxdepth, detail=True, **kwargs):
            if withdirs:
                files.update(dirs)
            out.update({info["name"]: info for name, info in files.items()})
        if not out and cls.isfile(path):
            # walk works on directories, but find should also return [path]
            # when path happens to be a file
            out[path] = {}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    @classmethod
    def ls(cls, path, detail=False, **kwargs):
        path = cls._strip_protocol(path)
        if detail:
            with os.scandir(path) as it:
                return [cls.info(f) for f in it]
        else:
            return [posixpath.join(path, f) for f in os.listdir(path)]

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        if path.startswith("file://"):
            path = path[7:]
        elif path.startswith("file:"):
            path = path[5:]
        return make_path_posix(path).rstrip("/") or cls.root_marker


def make_path_posix(path, sep=os.sep):
    """Make path generic"""
    if isinstance(path, (list, set, tuple)):
        return type(path)(make_path_posix(p) for p in path)
    if "~" in path:
        path = os.path.expanduser(path)
    if sep == "/":
        # most common fast case for posix
        if path.startswith("/"):
            return path
        if path.startswith("./"):
            path = path[2:]
        return os.getcwd() + "/" + path
    if (
        (sep not in path and "/" not in path)
        or (sep == "/" and not path.startswith("/"))
        or (sep == "\\" and ":" not in path and not path.startswith("\\\\"))
    ):
        # relative path like "path" or "rel\\path" (win) or rel/path"
        if os.sep == "\\":
            # abspath made some more '\\' separators
            return make_path_posix(os.path.abspath(path))
        else:
            return os.getcwd() + "/" + path
    if path.startswith("file://"):
        path = path[7:]
    if re.match("/[A-Za-z]:", path):
        # for windows file URI like "file:///C:/folder/file"
        # or "file:///C:\\dir\\file"
        path = path[1:].replace("\\", "/").replace("//", "/")
    if path.startswith("\\\\"):
        # special case for windows UNC/DFS-style paths, do nothing,
        # just flip the slashes around (case below does not work!)
        return path.replace("\\", "/")
    if re.match("[A-Za-z]:", path):
        # windows full path like "C:\\local\\path"
        return path.lstrip("\\").replace("\\", "/").replace("//", "/")
    if path.startswith("\\"):
        # windows network path like "\\server\\path"
        return "/" + path.lstrip("\\").replace("\\", "/").replace("//", "/")
    return path


def stringify_path(filepath):
    """Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath: object to be converted

    Returns
    -------
    filepath_str: maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced according to its
    __fspath__ method.
    For backwards compatibility with older Python version, pathlib.Path
    objects are specially coerced.
    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if isinstance(filepath, str):
        return filepath
    elif hasattr(filepath, "__fspath__"):
        return filepath.__fspath__()
    elif isinstance(filepath, Path):
        return str(filepath)
    elif hasattr(filepath, "path"):
        return filepath.path
    else:
        return filepath
