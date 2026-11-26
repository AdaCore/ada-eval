--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Infinite_Sequences.Higher_Order)
function Transform
  (S              : Sequence;
   Transform_Item :
     not null access function (E : Element_Type) return Element_Type)
   return Sequence is
begin
   return Res : Sequence do
      for I in Interval'(1, Length (S)) loop
         Res := Add (Res, Transform_Item (Get (S, I)));
      end loop;
   end return;
end Transform;
