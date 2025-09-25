with Ada.Assertions; use Ada.Assertions;

with Integer_Utils; use Integer_Utils;

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

   procedure Test_Absolute_Value is
      Result : Natural;
   begin
      Absolute_Value (1, Result);
      pragma Assert (Result = 1, "Expected 1, got " & Result'Image);
      Absolute_Value (-1, Result);
      pragma Assert (Result = 1, "Expected 1, got " & Result'Image);
      Absolute_Value (Integer'First + 1, Result);
      pragma
        Assert
          (Result = Integer'Last,
           "Expected " & Integer'Last'Image & ", got " & Result'Image);
   end Test_Absolute_Value;

begin
   Check_Assertions_Enabled;
   Test_Absolute_Value;
end Tests;
