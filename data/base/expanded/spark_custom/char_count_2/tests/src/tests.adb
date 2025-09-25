with Ada.Assertions; use Ada.Assertions;

with String_Utils; use String_Utils;

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

   procedure Test_Count is
   begin
      pragma
        Assert (Count ("", 'l') = 0, "Count should be 0 for empty string");
      pragma
        Assert (Count ("Hello world!", 'l') = 3, "Count of 'l' should be 3");
      pragma
        Assert
          (Count ("Hello world!", 'h') = 0, "Count should be case sensitive");
      pragma
        Assert
          (Count ("Hello world!", 'H') = 1, "Count should be case sensitive");
   end Test_Count;

begin
   Check_Assertions_Enabled;
   Test_Count;
end Tests;
