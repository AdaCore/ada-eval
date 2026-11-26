pragma Assertion_Policy (Ignore);

with Interfaces.C; use Interfaces.C;

package C_Strings
  with SPARK_Mode, Always_Terminates
is

   type chars_ptr is private
   with Default_Initial_Condition => Is_Null (Chars_Ptr);
   pragma Preelaborable_Initialization (chars_ptr);

   Null_Ptr : constant chars_ptr;

   --  We should start the copy here

   function Is_Null (Item : chars_ptr) return Boolean
   with Ghost, Post => Is_Null'Result = (Item = Null_Ptr);

   function C_Length_Ghost (Item : char_array) return size_t
   with
     Ghost,
     Import,
     Global   => null,
     Pre      => Is_Nul_Terminated (Item),
     Post     => C_Length_Ghost'Result = Interfaces.C.C_Length_Ghost (Item),
     Annotate => (GNATprove, Inline_For_Proof);

   --  SPARK primitives

   function New_Char_Array (Chars : char_array) return chars_ptr
   with
     --  `Volatile_Function,` upstream; dropped here
     --  The value of New_Char_Array'Result is different each time it is
     --  called.

     Pre    =>
       Chars'Last < size_t'Last
       or else Chars'First > 0
       or else Is_Nul_Terminated (Chars),
     --  This precondition ensures that Strlen does not wrap-around on the
     --  new array.

     Post   =>
       New_Char_Array'Result /= Null_Ptr
       and then
         Strlen (New_Char_Array'Result)
         = (if Is_Nul_Terminated (Chars)
            then C_Length_Ghost (Chars)
            else Chars'Length)
       --  Strlen returns the number of elements before the first occurrence of
       --  nul in Chars & nul.

       and then
         Value (New_Char_Array'Result) (Strlen (New_Char_Array'Result)) = nul
       and then
         (if Strlen (New_Char_Array'Result) > 0
          then
            (for all I in 0 .. Strlen (New_Char_Array'Result) - 1 =>
               Value (New_Char_Array'Result) (I) = Chars (Chars'First + I))),
     --  Value returns the prefix of Chars & nul up to and including the first
     --  occurrence of nul.

     Global => null;

   function Value (Item : chars_ptr) return char_array
   with
     Pre    => Item /= Null_Ptr,
     Post   =>
       Value'Result'First = 0
       and then Value'Result'Last = Strlen (Item)
       and then Value'Result (Strlen (Item)) = nul
       and then
         (for all I in 0 .. Strlen (Item) =>
            (if I < Strlen (Item) then Value'Result (I) /= nul)),
     Global => null;
   --  Value returns the prefix of the value pointed by Item up to and
   --  including the first occurrence of nul.

   function Strlen (Item : chars_ptr) return size_t
   with Pre => Item /= Null_Ptr, Global => null;
   --  Strlen returns the number of elements before the first occurrence of nul
   --  in the value pointed by Item.

private
   type Chars_Ptr is access char_array
   with
     Predicate =>
       Chars_Ptr = null
       or else (Chars_Ptr'First = 0 and then Chars_Ptr (Chars_Ptr'Last) = nul);
   --  To simplify proofs, the array is slided to 0 and truncated at the first
   --  Nul.

   Null_Ptr : constant Chars_Ptr := null;

   function Is_Null (Item : Chars_Ptr) return Boolean
   is (Item = null);
end C_Strings;
