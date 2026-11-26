pragma Assertion_Policy (Ignore);
separate (C_Strings)
function New_Char_Array (Chars : char_array) return Chars_Ptr is
   Found      : Boolean := False;
   First_Null : size_t
   with Relaxed_Initialization;
begin
   for I in Chars'Range loop
      if Chars (I) = nul then
         Found := True;
         First_Null := I;
         Lemma_Is_Find_Nul (Chars, First_Null);
         exit;
      end if;
      pragma
        Loop_Invariant (for all J in Chars'First .. I => Chars (J) /= nul);
   end loop;

   if Found then
      declare
         L_Res : constant size_t := First_Null - Chars'First;
         Res   : constant char_array (0 .. L_Res) :=
           Chars (Chars'First .. First_Null);
      begin
         Lemma_Is_Find_Nul (Res, L_Res);
         pragma Assert (L_Res = C_Length_Ghost (Chars));
         pragma
           Assert
             (for all I in 0 .. C_Length_Ghost (Chars) =>
                Res (I) = Chars (Chars'First + I));
         pragma
           Assert
             (Find_Nul (Res)
              = (if Is_Nul_Terminated (Chars)
                 then C_Length_Ghost (Chars)
                 else Chars'Length));
         return new char_array'(Res);
      end;
   elsif Chars'Length = 0 then
      declare
         Res : constant char_array := [0 => nul];
      begin
         pragma Assert (Find_Nul (Res) = 0);
         pragma
           Assert
             (Find_Nul (Res)
              = (if Is_Nul_Terminated (Chars)
                 then C_Length_Ghost (Chars)
                 else Chars'Length));
         return new char_array'(Res);
      end;
   else
      declare
         Left : constant char_array (0 .. Chars'Length - 1) := Chars;
         Res  : constant char_array := Left & nul;
      begin
         pragma Assert (for all I in Chars'Range => Chars (I) /= nul);
         pragma
           Assert
             (for all I in 0 .. size_t (Chars'Length - 1) =>
                By
                  (Res (I) = Chars (Chars'First + I),
                   Left (I) = Chars (Chars'First + I)));
         pragma
           Assert
             (for all I in 0 .. size_t (Chars'Length - 1) =>
                By
                  (Res (I) /= nul,
                   Res (I) = Chars (Chars'First + I)
                   and then
                     By
                       (Chars (Chars'First + I) /= nul,
                        Chars'First + I in Chars'Range)));
         pragma Assert (Res (Res'Last) = nul);
         Lemma_Is_Find_Nul (Res, Chars'Length);
         pragma
           Assert
             (Find_Nul (Res)
              = (if Is_Nul_Terminated (Chars)
                 then C_Length_Ghost (Chars)
                 else Chars'Length));
         return new char_array'(Res);
      end;
   end if;
end New_Char_Array;
