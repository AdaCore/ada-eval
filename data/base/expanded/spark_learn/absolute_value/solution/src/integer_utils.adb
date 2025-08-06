package Integer_Utils is

   procedure Absolute_Value (X : Integer; R : out Natural) is
   begin
      if X < 0 then
         R := -X;
      else
         R := X;
      end if;
   end Absolute_Value;

end Integer_Utils;
