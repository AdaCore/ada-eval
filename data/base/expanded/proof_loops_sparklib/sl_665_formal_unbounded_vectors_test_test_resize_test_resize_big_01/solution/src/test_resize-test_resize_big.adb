separate (Test_Resize)
function Test_Resize_Big return Vector is
   X : Vector;
begin
   for I in 1 .. 1000 loop
      Append (X, I);
      pragma Loop_Invariant (Length (X) = Count_Type (I));
   end loop;
   return X;
end Test_Resize_Big;
