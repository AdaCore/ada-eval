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

--  The actuals are upstream's own: Long_Integer and Long_Float, and the
--  default
--  Maximum_Number_Length of 30.  Upstream instantiates JSON.Tokenizers from
--  inside the body of JSON.Parsers (json-parsers.adb:28) with the Types
--  instance it was itself given, so instantiating it directly here reaches the
--  same instance without copying JSON.Parsers.

with JSON.Types;
with JSON.Tokenizers;

package Instances
  with SPARK_Mode => On
is

   package Types is new JSON.Types (Long_Integer, Long_Float);
   --  JSON.Types instantiated with the widest predefined numeric types, which
   --  is what the tests, the tools and the README example all use

   package Tokenizers is new JSON.Tokenizers (Types);

end Instances;
