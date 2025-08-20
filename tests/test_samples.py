import re
from pathlib import Path

import pytest

from ada_eval.datasets.types.samples import (
    Location,
    PathMustBeRelativeError,
    SubprogramNotFoundError,
)

TEST_ADB = """\
package body Some_Name is

   Some_Name : constant String := "Some_Value";

   function Some_Name_With_Suffix return string is
   begin
      return "Return Value";
   end Some_Name_With_Suffix;

   function Some_Name (I : Integer) return Integer is
   begin
      return I + 1;
   end Some_Name;

   procedure Some_Name is null;

end Some_Name;
"""


def test_location_absolute_path():
    error_msg = "Path '/absolute/path' is not relative"
    with pytest.raises(PathMustBeRelativeError, match=re.escape(error_msg)):
        Location(path=Path("/absolute/path"), subprogram_name="Some_Name")


def test_location_find_line_number(tmp_path: Path):
    # Write the test `.adb` file
    test_adb = tmp_path / "some_name.adb"
    test_adb.write_text(TEST_ADB)
    # Check that `find_line_number()` finds the first instance of `Some_Name`
    # as a subprogram (i.e. line 10)
    loc = Location(path=test_adb.relative_to(tmp_path), subprogram_name="Some_Name")
    assert loc.find_line_number(tmp_path) == 10
    # Check that it works from a different path
    (tmp_path / "src").mkdir()
    test_adb = test_adb.rename(tmp_path / "src" / "some_name.adb")
    loc = Location(path=test_adb.relative_to(tmp_path), subprogram_name="Some_Name")
    assert loc.find_line_number(tmp_path) == 10
    # Check that procedures are also recognised as subprograms
    new_content = re.sub(
        r"function Some_Name .*?end Some_Name;", "", TEST_ADB, flags=re.DOTALL
    )
    test_adb.write_text(new_content)
    assert loc.find_line_number(tmp_path) == 12  # 2 lines of residual whitespace
    # Check that it raises an error if no subprogram is found
    new_content = new_content.replace("procedure Some_Name is null;", "")
    test_adb.write_text(new_content)
    error_msg = f"Subprogram 'Some_Name' not found in '{test_adb}'"
    with pytest.raises(SubprogramNotFoundError, match=re.escape(error_msg)):
        loc.find_line_number(tmp_path)
