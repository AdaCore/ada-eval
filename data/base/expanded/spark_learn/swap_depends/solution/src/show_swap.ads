package Show_Swap is

   procedure Swap (X, Y : in out Positive)
   with Depends => (X => Y, Y => X);

   procedure Identity (X, Y : in out Positive)
   with Depends => (X => X, Y => Y);

end Show_Swap;
