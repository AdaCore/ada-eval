package body Integer_Utils is

   procedure Absolute (X : in out Integer) is
   begin
      if X > 0 then
         X := -X;
      end if;
   end Absolute;

end Integer_Utils;
