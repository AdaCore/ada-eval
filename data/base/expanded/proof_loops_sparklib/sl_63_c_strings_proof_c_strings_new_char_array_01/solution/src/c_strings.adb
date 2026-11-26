pragma Assertion_Policy (Ignore);

with SPARK.Cut_Operations; use SPARK.Cut_Operations;

package body C_Strings
  with SPARK_Mode
is

   -----------------------
   -- Local Subprograms --
   -----------------------

   function Find_Nul (Chars : char_array) return size_t
   with
     Import,
     Global => null,
     Pre    => Is_Nul_Terminated (Chars),
     Post   => Find_Nul'Result = Chars'First + C_Length_Ghost (Chars);

   procedure Lemma_Is_Find_Nul (Chars : char_array; P : size_t)
   with
     Ghost,
     Pre  =>
       P in Chars'Range
       and then Chars (P) = nul
       and then
         (for all I in Chars'First .. P => (if I < P then Chars (I) /= nul)),
     Post => Is_Nul_Terminated (Chars) and then P = Find_Nul (Chars);

   -----------------------
   -- Lemma_Is_Find_Nul --
   -----------------------

   procedure Lemma_Is_Find_Nul (Chars : char_array; P : size_t) is null;

   --------------------
   -- New_Char_Array --
   --------------------

   function New_Char_Array (Chars : char_array) return Chars_Ptr is separate;

   ------------
   -- Strlen --
   ------------

   function Strlen (Item : Chars_Ptr) return size_t
   is (Find_Nul (Item.all));

   -----------
   -- Value --
   -----------

   function Value (Item : Chars_Ptr) return char_array
   is (Item.all (0 .. Find_Nul (Item.all)));

end C_Strings;
