--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Sets.Higher_Order)
function Transform_Impl
  (S              : Set;
   Transform_Item :
     not null access function (E : Element_Type) return Element_Type)
   return Set
is
begin
   return Res : Set do
      for Subset in Iterate (S) loop
         declare
            E : constant Element_Type := Transform_Item (Choose (Subset));
         begin
            if not Contains (Res, E) then
               Res := Add (Res, E);
            end if;
         end;
      end loop;
   end return;
end Transform_Impl;
