--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Sets.Higher_Order)
function Filter
  (S    : Set;
   Test : not null access function (E : Element_Type) return Boolean)
   return Set is
begin
   return Res : Set do
      for Subset in Iterate (S) loop
         declare
            E : Element_Type renames Choose (Subset);
         begin
            if Test (E) then
               Res := Add (Res, E);
            end if;
         end;
      end loop;
   end return;
end Filter;
