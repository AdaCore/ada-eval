separate (Test)
function Test_Add return Sequence is
   X : Sequence;

begin
   --  Insertion at the end is optimized. Insert enough elements to trigger
   --  a resize.
   for I in 1 .. 150 loop
      X := Add (X, I);
      pragma Loop_Invariant (Last (X) = I);
      pragma Loop_Invariant (for all K in 1 .. I => Get (X, K) = K);
   end loop;
   return X;
end Test_Add;
