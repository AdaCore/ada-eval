with Ada.Assertions; use Ada.Assertions;
with Ada.Numerics.Discrete_Random;

with Array_Utils; use Array_Utils;

procedure Tests is

   procedure Check_Assertions_Enabled is
   begin
      begin
         pragma Assert (False, "Should raise");
      exception
         when others =>
            return; -- properly raised
      end;
      raise Program_Error with "Assertions not enabled";
   end Check_Assertions_Enabled;

   procedure Test_Update is
      A1 : Nat_Array := [1 => 1];
      A2 : Nat_Array (11 .. 20) := [for I in 1 .. 10 => I];
   begin
      Update (A1, 0, 1, 8, 4);
      pragma Assert (A1 = [2]);
      Update (A1, -1000, 1001, 245273732, 197324);
      pragma Assert (A1 = [1243]);
      Update (A2, -5, 20, 64, 4);
      pragma Assert (A2 = [1, 2, 3, 4, 16, 6, 7, 8, 9, 10]);
   end Test_Update;

begin
   Check_Assertions_Enabled;
   Test_Update;
end Tests;
