with Ada.Assertions; use Ada.Assertions;

with Placeholder; use Placeholder;

procedure Tests is

   function Candidate (Number : Float) return Float
   renames Placeholder.Truncate_Number;

begin
   pragma Assert (Candidate (3.5) = 0.5);
   pragma Assert (Candidate (1.25) = 0.25);
   pragma Assert (Candidate (123.0) = 0.0);
   pragma Assert (Candidate (Float'Last) = 0.0);
   pragma Assert (Candidate (Float'Model_Small) = Float'Model_Small);
end Tests;
