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
   end loop;

   if Found then
      declare
         L_Res : constant size_t := First_Null - Chars'First;
         Res   : constant char_array (0 .. L_Res) :=
           Chars (Chars'First .. First_Null);
      begin
         Lemma_Is_Find_Nul (Res, L_Res);
         return new char_array'(Res);
      end;
   elsif Chars'Length = 0 then
      declare
         Res : constant char_array := [0 => nul];
      begin
         return new char_array'(Res);
      end;
   else
      declare
         Left : constant char_array (0 .. Chars'Length - 1) := Chars;
         Res  : constant char_array := Left & nul;
      begin
         Lemma_Is_Find_Nul (Res, Chars'Length);
         return new char_array'(Res);
      end;
   end if;
end New_Char_Array;
