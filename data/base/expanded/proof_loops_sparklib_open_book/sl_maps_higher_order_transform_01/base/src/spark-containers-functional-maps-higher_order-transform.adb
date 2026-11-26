--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Maps.Higher_Order)
function Transform
  (M              : Map;
   Transform_Key  : not null access function (K : Key_Type) return Key_Type;
   Transform_Item :
     not null access function (E : Element_Type) return Element_Type)
   return Map is
begin
   return Res : Map do
      for Submap in Iterate (M) loop

         declare
            K : Key_Type renames Choose (Submap);
         begin
            Res :=
              Add
                (Res, Transform_Key (K), Transform_Item (Get (Submap, K)));
         end;
      end loop;
   end return;
end Transform;
