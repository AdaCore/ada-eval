package body Show_Swap is

   procedure Swap (X, Y : in out Positive) is
      Tmp : constant Positive := X;
   begin
      X := Y;
      Y := Tmp;
   end Swap;

   procedure Identity (X, Y : in out Positive) is
   begin
      Swap (X, Y);
      pragma Warnings (Off, "actuals for this call may be in wrong order");
      Swap (Y, X);
      pragma Warnings (On, "actuals for this call may be in wrong order");
   end Identity;

end Show_Swap;
