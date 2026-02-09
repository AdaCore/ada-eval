with Ada.Assertions; use Ada.Assertions;

with Placeholder; use Placeholder;
with Types;       use Types;

procedure Tests is

   function X (Operations : Integer_Array) return Boolean
   renames Placeholder.Below_Zero;

begin
   pragma Assert (X ([]) = False);
   pragma Assert (X ([1, 2, -3, 1, 2, -3]) = False);
   pragma Assert (X ([1, 2, -4, 5, 6]) = True);
   pragma Assert (X ([1, -1, 2, -2, 5, -5, 4, -4]) = False);
   pragma Assert (X ([1, -1, 2, -2, 5, -5, 4, -5]) = True);
   pragma Assert (X ([1, -2, 2, -2, 5, -5, 4, -4]) = True);
   pragma Assert (X ([Integer'First]) = True);
   pragma Assert (X ([1, Integer'Last, Integer'First]) = False);
   pragma Assert (X ([1, Integer'Last, Integer'First, Integer'First]) = True);
end Tests;
