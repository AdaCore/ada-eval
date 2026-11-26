--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Infinite_Sequences.Higher_Order)
function Sum_Impl
  (S     : Sequence;
   Last  : Big_Natural;
   Value : not null access function (E : Element_Type) return Big_Integer)
   return Big_Integer
is
begin
   return Res : Big_Integer := 0 do
      for I in Interval'(1, Last) loop
         Res := Res + Value (Get (S, I));
         pragma Loop_Invariant (Static => Res = Sum_Rec (S, I, Value));
      end loop;
   end return;
end Sum_Impl;
