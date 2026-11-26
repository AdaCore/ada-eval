--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Move (Target : in out List; Source : in out List) is
   N    : Node_Array renames Source.Nodes;
   X    : Count_Type;
   Orig : constant Count_Type := Source.Length
   with Ghost => Static;

begin
   Clear (Target);

   --  Move Source's elements one at a time: append the first to Target,
   --  then unlink and free it in Source (Unlink_First_Node + Free). The
   --  running count Target.Length + Source.Length stays equal to the
   --  original Source length, so Target never overflows.

   while Source.Length > 1 loop
      pragma Loop_Invariant (Static => Structural_Invariant (Target));
      pragma Loop_Invariant (Static => Structural_Invariant (Source));
      pragma
        Loop_Invariant (Static => Target.Length + Source.Length = Orig);
      pragma Loop_Variant (Static => (Decreases => Source.Length));

      X := Source.First;
      Append (Target, N (X).Element);
      Unlink_First_Node (Source);
      Source.Length := Source.Length - 1;
      Free (Source, X);
   end loop;

   if Source.Length = 1 then

      --  Copy the last element, then drop the sole node

      X := Source.First;
      Append (Target, N (X).Element);
      Unlink_Sole_Node (Source);
      Source.Length := Source.Length - 1;
      Free (Source, X);
   end if;
end Move;
