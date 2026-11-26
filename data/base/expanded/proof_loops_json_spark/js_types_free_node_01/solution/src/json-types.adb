--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

--  Big_Integers is an Ada 2022 standard unit; silence GNAT 16's advisory
--  (-gnatwi) about referencing it from this library's earlier language mode.
pragma Warnings (Off, "* is an Ada 2022 unit");
with Ada.Numerics.Big_Numbers.Big_Integers;
pragma Warnings (On, "* is an Ada 2022 unit");
with Ada.Unchecked_Deallocation;

package body JSON.Types
  with SPARK_Mode => On
is

   use Ada.Numerics.Big_Numbers.Big_Integers;

   procedure Free_Text is new
     Ada.Unchecked_Deallocation
       (Object => String,
        Name   => Streams.Text_Access);

   procedure Dealloc is new
     Ada.Unchecked_Deallocation
       (Object => JSON_Value,
        Name   => JSON_Value_Access);

   function Size (V : access constant JSON_Value) return Big_Natural
   is (if V = null
       then To_Big_Integer (0)
       else To_Big_Integer (1) + Size (V.First_Child) + Size (V.Next))
   with Ghost, Subprogram_Variant => (Structural => V);
   --  Number of nodes reachable from V, following both children and
   --  siblings: the measure showing that freeing and reversing terminate.
   --  The SPARK ownership model guarantees the tree is acyclic and
   --  finite, so the recursion is well defined; its own termination is
   --  established by the structural variant.

   --  Free_Node carries the Subprogram_Variant that Free cannot: the
   --  variant would have to appear on Free's declaration in the spec,
   --  where the ghost Size measure is not visible

   procedure Free_Node (Object : in out JSON_Value_Access)
   with
     Post               => Object = null,
     Always_Terminates,
     Subprogram_Variant => (Decreases => Size (Object)),
     Depends            => (Object => null, null => Object);

   procedure Free_Node (Object : in out JSON_Value_Access) is separate;

   procedure Free (Object : in out JSON_Value_Access) is
   begin
      Free_Node (Object);
   end Free;

end JSON.Types;
