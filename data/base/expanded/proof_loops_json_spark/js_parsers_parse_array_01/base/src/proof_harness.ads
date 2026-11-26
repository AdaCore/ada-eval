--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2026 M. Anthony Aiello
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

with JSON.Parsers;
with JSON.Types;

--  The json library consists of generic packages, which GNATprove only
--  analyzes through instantiations. This package instantiates them with
--  the types used by the tests and tools, so that running GNATprove on
--  main.gpr generates and discharges their proof obligations.

package Proof_Harness
  with SPARK_Mode => On
is

   package Types is new JSON.Types (Long_Integer, Long_Float);
   --  JSON.Types instantiated with the widest predefined numeric types, which
   --  is what the tests, the tools and the README example all use

   package Parsers is new JSON.Parsers (Types);
   --  JSON.Parsers in its default configuration: duplicate object keys are
   --  accepted

end Proof_Harness;
