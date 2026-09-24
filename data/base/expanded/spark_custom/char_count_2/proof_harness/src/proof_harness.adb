with String_Utils;

package body Proof_Harness is

   procedure Lemma_Ghost_Is_Ref (Str : String; Char : Character; Idx : Integer)
   with
     Ghost,
     Subprogram_Variant => (Decreases => Idx),
     Post               =>
       String_Utils.Count_Ghost (Str, Char, Idx) = Ref_Count (Str, Char, Idx);

   procedure Lemma_Ghost_Is_Ref (Str : String; Char : Character; Idx : Integer)
   is
   begin
      if Idx in Str'Range then
         Lemma_Ghost_Is_Ref (Str, Char, Idx - 1);
      end if;
   end Lemma_Ghost_Is_Ref;

   function Check_Count (Str : String; Char : Character) return Natural is
      Result : constant Natural := String_Utils.Count (Str, Char);
   begin
      Lemma_Ghost_Is_Ref (Str, Char, Str'Last);
      return Result;
   end Check_Count;

end Proof_Harness;
