separate (Test_Higher_Order2)
procedure Prove_No_Overflows (A : Small_Array) is
   S : Integer := 0;
begin
   for I in A'Range loop
      pragma
        Loop_Invariant
          (if My_Sum_Int.No_Overflows (A, S, I)
           then My_Sum_Int.No_Overflows (A, 0, A'First));
      pragma
        Loop_Invariant
          (S
           in (I - A'First) * My_Small'First
            .. (I - A'First) * My_Small'Last);
      S := S + Value (A (I));
   end loop;
end Prove_No_Overflows;
