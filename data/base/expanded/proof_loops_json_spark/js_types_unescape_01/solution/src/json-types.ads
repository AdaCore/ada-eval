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

package JSON.Types
  with SPARK_Mode => On
is

   pragma Preelaborate;

   function Unescape (Text : String) return String
   with Post => Unescape'Result'Length <= Text'Length;
   --  Return Text with JSON escape sequences decoded; \uXXXX escape
   --  sequences (including UTF-16 surrogate pairs) are decoded to UTF-8
   --
   --  Upstream this function is declared in the package body and reached
   --  only through the expression functions Value and Key.  Promoting it
   --  to the spec is what makes GNATprove analyse it as a subprogram of
   --  its own here; upstream analyses it separately anyway, because the
   --  expression functions that call it are the package's own contracts.

end JSON.Types;
