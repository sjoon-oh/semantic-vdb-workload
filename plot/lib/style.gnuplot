#!/usr/bin/env gnuplot

# ==============================================================================
# Named color palettes
#
# Select a palette with:
#   -e "arg_color_palette='blue_gold'"
#
# The palettes are curated for multi-series plots. Several sets are adapted from
# color combinations collected at https://colorpalette.org/ and reordered so
# adjacent plot series are easier to distinguish (that site currently blocks
# automated fetches behind a Cloudflare challenge; the 20 palettes from
# desert_canyon through deep_sea below are hand-curated in the same style and
# format, not scraped from a specific colorpalette.org page).
#
# Backward-compatible aliases palette1 ... palette5 are retained.
# ==============================================================================

array COLOR_PALETTE[10]

# Muted cool colors; default.
if (arg_color_palette eq "research_muted") {
    COLOR_PALETTE[1] = "#817988"  # muted purple
    COLOR_PALETTE[2] = "#98adc9"  # muted blue
    COLOR_PALETTE[3] = "#595163"  # muted dark purple
    COLOR_PALETTE[4] = "#44667f"  # muted slate blue
    COLOR_PALETTE[5] = "#b2a5b3"  # muted lavender
    COLOR_PALETTE[6] = "#9095b0"  # muted gray-blue
    COLOR_PALETTE[7] = "#30303e"  # muted charcoal
    COLOR_PALETTE[8] = "#053348"  # muted navy
    COLOR_PALETTE[9] = "#34576b"  # muted steel blue
    COLOR_PALETTE[10] = "#ebe1e7"  # muted light
}
else if (arg_color_palette eq "blue_gold") {
    # Blue, light blue, gold, green, brown, and neutral accents.
    COLOR_PALETTE[1] = "#2b668b"  # blue
    COLOR_PALETTE[2] = "#c7a04d"  # gold
    COLOR_PALETTE[3] = "#64a3cd"  # light blue
    COLOR_PALETTE[4] = "#719095"  # muted green
    COLOR_PALETTE[5] = "#69501f"  # brown
    COLOR_PALETTE[6] = "#d0af87"  # light brown
    COLOR_PALETTE[7] = "#222513"  # dark green
    COLOR_PALETTE[8] = "#696f26"  # olive
    COLOR_PALETTE[9] = "#98adc1"  # light gray-blue
    COLOR_PALETTE[10] = "#ced6de"  # light gray
}
else if (arg_color_palette eq "slate_blue") {
    # Cool slate and powder-blue palette.
    COLOR_PALETTE[1] = "#546a84"  # slate
    COLOR_PALETTE[2] = "#a6a27c"  # powder blue
    COLOR_PALETTE[3] = "#7698af"  # light slate
    COLOR_PALETTE[4] = "#404c62"  # dark slate
    COLOR_PALETTE[5] = "#7c7d5d"  # muted green
    COLOR_PALETTE[6] = "#7aa4be"  # light blue
    COLOR_PALETTE[7] = "#a29d68"  # muted yellow
    COLOR_PALETTE[8] = "#a0c2d4"  # light gray-blue
    COLOR_PALETTE[9] = "#cce6f1"  # light blue
    COLOR_PALETTE[10] = "#e9f4f7"  # very light blue
}
else if (arg_color_palette eq "earth_teal") {
    # Earth, teal, orange, and gold.
    COLOR_PALETTE[1] = "#32977d"  # teal
    COLOR_PALETTE[2] = "#eb8b1e"  # orange
    COLOR_PALETTE[3] = "#7b3e12"  # brown
    COLOR_PALETTE[4] = "#5084a8"  # slate blue
    COLOR_PALETTE[5] = "#886039"  # muted gold
    COLOR_PALETTE[6] = "#fbce4a"  # gold
    COLOR_PALETTE[7] = "#3f2210"  # dark brown
    COLOR_PALETTE[8] = "#e5a160"  # light brown
    COLOR_PALETTE[9] = "#bebbc4"  # light gray
    COLOR_PALETTE[10] = "#e1b999"  # light tan
}
else if (arg_color_palette eq "deep_blue") {
    # Deep blue with magenta and olive accents.
    COLOR_PALETTE[1] = "#2555c7"  # deep blue
    COLOR_PALETTE[2] = "#a93965"  # magenta
    COLOR_PALETTE[3] = "#51a3dc"  # light blue
    COLOR_PALETTE[4] = "#c1a745"  # gold
    COLOR_PALETTE[5] = "#263f83"  # muted blue
    COLOR_PALETTE[6] = "#574655"  # muted purple
    COLOR_PALETTE[7] = "#0b1a5b"  # dark blue
    COLOR_PALETTE[8] = "#998ba7"  # muted lavender
    COLOR_PALETTE[9] = "#04040b"  # very dark
    COLOR_PALETTE[10] = "#c5e3ee"  # light
}
else if (arg_color_palette eq "orange_navy") {
    # Orange, navy, steel blue, and warm neutrals.
    COLOR_PALETTE[1] = "#f58d33"  # orange
    COLOR_PALETTE[2] = "#3d5565"  # navy
    COLOR_PALETTE[3] = "#d27f67"  # steel blue
    COLOR_PALETTE[4] = "#041a3f"  # dark blue
    COLOR_PALETTE[5] = "#ee9e59"  # light orange
    COLOR_PALETTE[6] = "#6c8192"  # light gray-blue
    COLOR_PALETTE[7] = "#5d4e36"  # dark brown
    COLOR_PALETTE[8] = "#e6ad73"  # light brown
    COLOR_PALETTE[9] = "#2f2418"  # dark gray
    COLOR_PALETTE[10] = "#d2ba9d"  # light tan
}
else if (arg_color_palette eq "forest_cottage") {
    # Forest/cottage colors adapted from a colorpalette.org collection.
    COLOR_PALETTE[1] = "#593029"  # dark brown
    COLOR_PALETTE[2] = "#b7cb62"  # light green
    COLOR_PALETTE[3] = "#8e665c"  # muted brown
    COLOR_PALETTE[4] = "#525632"  # dark green
    COLOR_PALETTE[5] = "#e3c08f"  # light tan
    COLOR_PALETTE[6] = "#8b8a10"  # olive
    COLOR_PALETTE[7] = "#ba836a"  # light brown
    COLOR_PALETTE[8] = "#b5a763"  # muted yellow
    COLOR_PALETTE[9] = "#b4a39b"  # light gray
    COLOR_PALETTE[10] = "#e3dad3"  # light pink
}
else if (arg_color_palette eq "ocean_mist") {
    # Ocean and mist colors.
    COLOR_PALETTE[1] = "#0b3954"  # dark blue
    COLOR_PALETTE[2] = "#087e8b"  # teal
    COLOR_PALETTE[3] = "#ff5a5f"  # coral
    COLOR_PALETTE[4] = "#5c9ead"  # ocean blue
    COLOR_PALETTE[5] = "#334e68"  # navy
    COLOR_PALETTE[6] = "#8ab6d6"  # light blue
    COLOR_PALETTE[7] = "#243b53"  # dark gray-blue
    COLOR_PALETTE[8] = "#9fb3c8"  # light gray-blue
    COLOR_PALETTE[9] = "#bfd7ea"  # very light blue
    COLOR_PALETTE[10] = "#d9e2ec"  # light gray
}
else if (arg_color_palette eq "sunset") {
    # Warm sunset colors.
    COLOR_PALETTE[1] = "#7a1f5b"  # deep red
    COLOR_PALETTE[2] = "#f95738"  # orange
    COLOR_PALETTE[3] = "#edae49"  # golden
    COLOR_PALETTE[4] = "#2e294e"  # deep blue
    COLOR_PALETTE[5] = "#d1495b"  # rose
    COLOR_PALETTE[6] = "#ee964b"  # light orange
    COLOR_PALETTE[7] = "#5d2e46"  # dark purple
    COLOR_PALETTE[8] = "#f4d35e"  # yellow
    COLOR_PALETTE[9] = "#a63d40"  # burgundy
    COLOR_PALETTE[10] = "#c97c5d"  # tan
}
else if (arg_color_palette eq "pastel") {
    # Soft pastel colors.
    COLOR_PALETTE[1] = "#6d8ea0"  # soft blue
    COLOR_PALETTE[2] = "#f2b5d4"  # soft pink
    COLOR_PALETTE[3] = "#9bbf8a"  # soft green
    COLOR_PALETTE[4] = "#c7a6d8"  # soft purple
    COLOR_PALETTE[5] = "#d99b73"  # soft orange
    COLOR_PALETTE[6] = "#8fb9aa"  # soft teal
    COLOR_PALETTE[7] = "#b6d8f2"  # soft light blue
    COLOR_PALETTE[8] = "#f7d6e0"  # soft light pink
    COLOR_PALETTE[9] = "#c7ceea"  # soft light purple
    COLOR_PALETTE[10] = "#ffdac1"  # soft light orange
}
else if (arg_color_palette eq "high_contrast") {
    # High-contrast categorical colors, suitable when series separation matters.
    COLOR_PALETTE[1] = "#0072b2"  # blue
    COLOR_PALETTE[2] = "#d55e00"  # orange
    COLOR_PALETTE[3] = "#009e73"  # green
    COLOR_PALETTE[4] = "#cc79a7"  # magenta
    COLOR_PALETTE[5] = "#e69f00"  # yellow
    COLOR_PALETTE[6] = "#56b4e9"  # light blue
    COLOR_PALETTE[7] = "#332288"  # dark blue
    COLOR_PALETTE[8] = "#000000"  # black
    COLOR_PALETTE[9] = "#999999"  # gray
    COLOR_PALETTE[10] = "#f0e442"  # light yellow
}
else if (arg_color_palette eq "mono_blue") {
    # Sequential blue shades.
    COLOR_PALETTE[1] = "#08306b"  # dark blue
    COLOR_PALETTE[2] = "#08519c"  # medium blue
    COLOR_PALETTE[3] = "#2171b5"  # light blue
    COLOR_PALETTE[4] = "#4292c6"  # lighter blue
    COLOR_PALETTE[5] = "#6baed6"  # lightest blue
    COLOR_PALETTE[6] = "#9ecae1"  # very light blue
    COLOR_PALETTE[7] = "#c6dbef"  # extremely light blue
    COLOR_PALETTE[8] = "#deebf7"  # almost white blue
    COLOR_PALETTE[9] = "#eff3ff"  # very almost white blue
    COLOR_PALETTE[10] = "#f7fbff"  # white blue
}
else if (arg_color_palette eq "terrace_field") {
    # Terrace Field Hill Station, colorpalette.org:
    # https://colorpalette.org/terrace-field-hill-station-color-palette/
    # Agricultural greens with yellow and cool-water accents.
    COLOR_PALETTE[1] = "#4a532d"  # dark olive green
    COLOR_PALETTE[2] = "#629893"  # sea green
    COLOR_PALETTE[3] = "#d5cf39"  # yellow green
    COLOR_PALETTE[4] = "#6c762f"  # olive
    COLOR_PALETTE[5] = "#7ebfb1"  # turquoise
    COLOR_PALETTE[6] = "#a9a831"  # muted yellow
    COLOR_PALETTE[7] = "#202719"  # dark forest
    COLOR_PALETTE[8] = "#7ba59a"  # muted teal
    COLOR_PALETTE[9] = "#99d2bf"  # pale teal
    COLOR_PALETTE[10] = "#d9f3d8"  # pale green
}
else if (arg_color_palette eq "cityscape") {
    # Metropolitan Area Cityscape City, colorpalette.org:
    # https://colorpalette.org/metropolitan-area-cityscape-city-color-palette-40/
    # Dark teal foundations with cyan and red accents.
    COLOR_PALETTE[1] = "#0b5b6b"  # deep teal
    COLOR_PALETTE[2] = "#bb0f26"  # crimson
    COLOR_PALETTE[3] = "#06aab9"  # bright cyan
    COLOR_PALETTE[4] = "#d15a3c"  # warm orange-red
    COLOR_PALETTE[5] = "#05303d"  # dark blue-teal
    COLOR_PALETTE[6] = "#61bec3"  # light turquoise
    COLOR_PALETTE[7] = "#571524"  # dark claret
    COLOR_PALETTE[8] = "#ae6c7d"  # muted rose
    COLOR_PALETTE[9] = "#044255"  # navy
    COLOR_PALETTE[10] = "#d3d5d7"  # cool gray
}
else if (arg_color_palette eq "winter_sky") {
    # Winter Sky Nature, colorpalette.org:
    # https://colorpalette.org/winter-sky-nature-color-palette/
    # A blue-forward sequential palette with enough contrast for line charts.
    COLOR_PALETTE[1] = "#0a4298"  # royal blue
    COLOR_PALETTE[2] = "#077ec7"  # clear blue
    COLOR_PALETTE[3] = "#3c6291"  # steel blue
    COLOR_PALETTE[4] = "#318fce"  # sky blue
    COLOR_PALETTE[5] = "#2b4269"  # dark slate blue
    COLOR_PALETTE[6] = "#4c95cd"  # medium blue
    COLOR_PALETTE[7] = "#689fcd"  # muted blue
    COLOR_PALETTE[8] = "#91b8dc"  # powder blue
    COLOR_PALETTE[9] = "#a7cce8"  # pale blue
    COLOR_PALETTE[10] = "#d2e3f0"  # ice blue
}
else if (arg_color_palette eq "woodland_autumn") {
    # Nature Woodland Autumn, colorpalette.org:
    # https://colorpalette.org/nature-woodland-autumn-color-palette/
    # High-contrast burnt orange/brown accents over warm neutrals.
    COLOR_PALETTE[1] = "#ba3d20"  # burnt red
    COLOR_PALETTE[2] = "#d37b2b"  # autumn orange
    COLOR_PALETTE[3] = "#5d513c"  # olive brown
    COLOR_PALETTE[4] = "#823324"  # dark rust
    COLOR_PALETTE[5] = "#532220"  # deep brown
    COLOR_PALETTE[6] = "#d79655"  # tan orange
    COLOR_PALETTE[7] = "#9d9071"  # muted khaki
    COLOR_PALETTE[8] = "#8e7155"  # brown gray
    COLOR_PALETTE[9] = "#bc8e7e"  # dusty rose
    COLOR_PALETTE[10] = "#f7ead6"  # warm off-white
}
else if (arg_color_palette eq "desert_canyon") {
    # Curated: red-rock canyon terracotta and sand with a dusty-sky-blue accent.
    COLOR_PALETTE[1] = "#b5482c"  # burnt sienna
    COLOR_PALETTE[2] = "#e8a33d"  # ochre
    COLOR_PALETTE[3] = "#8a2f1f"  # dark rust
    COLOR_PALETTE[4] = "#6f8c96"  # dusty blue
    COLOR_PALETTE[5] = "#d9764a"  # terracotta
    COLOR_PALETTE[6] = "#f2c879"  # sand
    COLOR_PALETTE[7] = "#4a2818"  # dark umber
    COLOR_PALETTE[8] = "#c9b79c"  # pale sand
    COLOR_PALETTE[9] = "#3d5a63"  # deep dusty blue
    COLOR_PALETTE[10] = "#f6e4c8"  # cream
}
else if (arg_color_palette eq "arctic_ice") {
    # Curated: glacier blues from near-black polar night to snow white.
    COLOR_PALETTE[1] = "#1c4e6b"  # deep glacier blue
    COLOR_PALETTE[2] = "#dff0f5"  # ice white
    COLOR_PALETTE[3] = "#3f8fae"  # glacier blue
    COLOR_PALETTE[4] = "#0a2740"  # polar navy
    COLOR_PALETTE[5] = "#8fc7db"  # pale ice
    COLOR_PALETTE[6] = "#c8e6ee"  # frost
    COLOR_PALETTE[7] = "#274b5e"  # slate ice
    COLOR_PALETTE[8] = "#5fa9c4"  # mid glacier
    COLOR_PALETTE[9] = "#0e1c26"  # arctic night
    COLOR_PALETTE[10] = "#eef7fa"  # snow white
}
else if (arg_color_palette eq "volcanic_ash") {
    # Curated: charcoal and ash grays with molten-ember red/orange accents.
    COLOR_PALETTE[1] = "#2b2b2e"  # charcoal ash
    COLOR_PALETTE[2] = "#c0392b"  # molten red
    COLOR_PALETTE[3] = "#57534f"  # gray ash
    COLOR_PALETTE[4] = "#e67e22"  # ember orange
    COLOR_PALETTE[5] = "#171716"  # obsidian
    COLOR_PALETTE[6] = "#8c7d78"  # dusty ash
    COLOR_PALETTE[7] = "#8e2311"  # dark ember
    COLOR_PALETTE[8] = "#a89f9a"  # pale ash
    COLOR_PALETTE[9] = "#3f1208"  # scorched
    COLOR_PALETTE[10] = "#d9c9c2"  # ash white
}
else if (arg_color_palette eq "coral_reef") {
    # Curated: reef teal and turquoise against warm coral and shell pink.
    COLOR_PALETTE[1] = "#0e7c7b"  # reef teal
    COLOR_PALETTE[2] = "#ff6f61"  # coral
    COLOR_PALETTE[3] = "#2fb8a6"  # turquoise
    COLOR_PALETTE[4] = "#ffab91"  # pale coral
    COLOR_PALETTE[5] = "#0a4d4c"  # deep teal
    COLOR_PALETTE[6] = "#7fd8c9"  # seafoam
    COLOR_PALETTE[7] = "#d1495b"  # rose coral
    COLOR_PALETTE[8] = "#a8e6df"  # pale seafoam
    COLOR_PALETTE[9] = "#053937"  # abyss teal
    COLOR_PALETTE[10] = "#ffe0d4"  # shell pink
}
else if (arg_color_palette eq "autumn_maple") {
    # Curated: maple-leaf reds and oranges over dark bark and golden yellow.
    COLOR_PALETTE[1] = "#8c1c13"  # maple red
    COLOR_PALETTE[2] = "#e0812b"  # maple orange
    COLOR_PALETTE[3] = "#5c1009"  # dark maple
    COLOR_PALETTE[4] = "#f2b632"  # golden yellow
    COLOR_PALETTE[5] = "#b23a1f"  # burnt orange
    COLOR_PALETTE[6] = "#6b3418"  # bark brown
    COLOR_PALETTE[7] = "#f4d58d"  # pale gold
    COLOR_PALETTE[8] = "#3c1a0e"  # dark bark
    COLOR_PALETTE[9] = "#d97a3e"  # amber
    COLOR_PALETTE[10] = "#f6e6c3"  # cream
}
else if (arg_color_palette eq "lavender_dusk") {
    # Curated: violet and lavender through dusk pink and steel blue.
    COLOR_PALETTE[1] = "#5b4b8a"  # lavender purple
    COLOR_PALETTE[2] = "#e8a0bf"  # dusk pink
    COLOR_PALETTE[3] = "#7d6bb0"  # light lavender
    COLOR_PALETTE[4] = "#37285c"  # deep violet
    COLOR_PALETTE[5] = "#c48fc7"  # orchid
    COLOR_PALETTE[6] = "#a6c6e0"  # dusk blue
    COLOR_PALETTE[7] = "#231a3f"  # midnight violet
    COLOR_PALETTE[8] = "#f0c9dc"  # pale pink
    COLOR_PALETTE[9] = "#4a6fa0"  # dusk steel blue
    COLOR_PALETTE[10] = "#f5ecf5"  # pale lilac
}
else if (arg_color_palette eq "midnight_amber") {
    # Curated: ink-black and deep navy with warm amber and bronze accents.
    COLOR_PALETTE[1] = "#0d1b2a"  # midnight navy
    COLOR_PALETTE[2] = "#e0a458"  # amber
    COLOR_PALETTE[3] = "#1b3a5c"  # deep navy
    COLOR_PALETTE[4] = "#f4c77a"  # light amber
    COLOR_PALETTE[5] = "#050b12"  # ink black
    COLOR_PALETTE[6] = "#8a5a2b"  # bronze
    COLOR_PALETTE[7] = "#3d5a80"  # steel blue
    COLOR_PALETTE[8] = "#c98a3f"  # burnished gold
    COLOR_PALETTE[9] = "#8fa8c2"  # pale steel blue
    COLOR_PALETTE[10] = "#f0dcb5"  # cream gold
}
else if (arg_color_palette eq "citrus_grove") {
    # Curated: leaf and lime greens with lemon yellow and orange citrus accents.
    COLOR_PALETTE[1] = "#3f7d20"  # citrus leaf green
    COLOR_PALETTE[2] = "#f5c518"  # lemon yellow
    COLOR_PALETTE[3] = "#6fae2e"  # grove green
    COLOR_PALETTE[4] = "#f28c28"  # orange
    COLOR_PALETTE[5] = "#274d0f"  # dark leaf
    COLOR_PALETTE[6] = "#f7e07e"  # pale lemon
    COLOR_PALETTE[7] = "#a5d75f"  # lime
    COLOR_PALETTE[8] = "#b9591f"  # burnt orange
    COLOR_PALETTE[9] = "#1b330a"  # deep forest
    COLOR_PALETTE[10] = "#fdf3c4"  # pale citrus
}
else if (arg_color_palette eq "rust_copper") {
    # Curated: copper and rust metals from espresso-dark to pale tan.
    COLOR_PALETTE[1] = "#7a3b1e"  # copper brown
    COLOR_PALETTE[2] = "#c97b3d"  # rust orange
    COLOR_PALETTE[3] = "#4d230f"  # dark copper
    COLOR_PALETTE[4] = "#e6a35c"  # light rust
    COLOR_PALETTE[5] = "#2c1408"  # espresso
    COLOR_PALETTE[6] = "#9a5a2e"  # burnt copper
    COLOR_PALETTE[7] = "#f0c48f"  # pale copper
    COLOR_PALETTE[8] = "#5e3319"  # dark rust brown
    COLOR_PALETTE[9] = "#d98c4a"  # amber copper
    COLOR_PALETTE[10] = "#f7e3c9"  # cream tan
}
else if (arg_color_palette eq "spring_meadow") {
    # Curated: fresh meadow greens with buttercup yellow and pale chartreuse.
    COLOR_PALETTE[1] = "#4f9d4a"  # meadow green
    COLOR_PALETTE[2] = "#f4e06d"  # buttercup
    COLOR_PALETTE[3] = "#7fc76f"  # fresh green
    COLOR_PALETTE[4] = "#e8f28c"  # pale chartreuse
    COLOR_PALETTE[5] = "#2e6b2b"  # deep green
    COLOR_PALETTE[6] = "#d8f0a0"  # pale meadow
    COLOR_PALETTE[7] = "#a3d977"  # spring green
    COLOR_PALETTE[8] = "#f7f2d9"  # cream white
    COLOR_PALETTE[9] = "#1c4519"  # forest shadow
    COLOR_PALETTE[10] = "#c4e88a"  # light lime
}
else if (arg_color_palette eq "plum_wine") {
    # Curated: deep plum and wine through berry pink and dusty rose.
    COLOR_PALETTE[1] = "#5c1a3b"  # plum
    COLOR_PALETTE[2] = "#a13a5c"  # wine rose
    COLOR_PALETTE[3] = "#7a2650"  # deep plum
    COLOR_PALETTE[4] = "#d97b98"  # dusty rose
    COLOR_PALETTE[5] = "#3a0f24"  # dark plum
    COLOR_PALETTE[6] = "#c14f74"  # berry pink
    COLOR_PALETTE[7] = "#e8a7bd"  # pale rose
    COLOR_PALETTE[8] = "#2a0a17"  # wine black
    COLOR_PALETTE[9] = "#8c3a5c"  # muted mauve
    COLOR_PALETTE[10] = "#f2d6e0"  # pale pink
}
else if (arg_color_palette eq "steel_industrial") {
    # Curated: gunmetal and steel grays with a burnt-rust warm accent.
    COLOR_PALETTE[1] = "#3d4a52"  # gunmetal
    COLOR_PALETTE[2] = "#c2622d"  # rust accent
    COLOR_PALETTE[3] = "#5f6f78"  # steel gray
    COLOR_PALETTE[4] = "#e8845a"  # burnt orange accent
    COLOR_PALETTE[5] = "#22282c"  # dark charcoal
    COLOR_PALETTE[6] = "#8c99a1"  # light steel
    COLOR_PALETTE[7] = "#9a3d1a"  # dark rust
    COLOR_PALETTE[8] = "#b6c0c6"  # pale steel
    COLOR_PALETTE[9] = "#141719"  # near black
    COLOR_PALETTE[10] = "#f0b590"  # light rust accent
}
else if (arg_color_palette eq "tropical_bloom") {
    # Curated: hot pink and magenta against turquoise and tropical yellow.
    COLOR_PALETTE[1] = "#e6398a"  # hot pink
    COLOR_PALETTE[2] = "#12a5a0"  # turquoise
    COLOR_PALETTE[3] = "#f2a5cb"  # pale pink
    COLOR_PALETTE[4] = "#0b6e6b"  # deep turquoise
    COLOR_PALETTE[5] = "#f7d33c"  # tropical yellow
    COLOR_PALETTE[6] = "#8c1a5c"  # magenta
    COLOR_PALETTE[7] = "#7de0d9"  # pale turquoise
    COLOR_PALETTE[8] = "#053836"  # deep teal
    COLOR_PALETTE[9] = "#fce997"  # pale yellow
    COLOR_PALETTE[10] = "#c9127a"  # deep pink
}
else if (arg_color_palette eq "sandstone") {
    # Curated: sandstone tan and terracotta neutrals with a cool sage accent.
    COLOR_PALETTE[1] = "#c9a06a"  # sandstone tan
    COLOR_PALETTE[2] = "#8a5a3c"  # terracotta brown
    COLOR_PALETTE[3] = "#e8cba0"  # pale sand
    COLOR_PALETTE[4] = "#5c3a24"  # dark earth
    COLOR_PALETTE[5] = "#a3c9a5"  # sage accent
    COLOR_PALETTE[6] = "#f2e2c4"  # cream sand
    COLOR_PALETTE[7] = "#6b8f6d"  # deep sage
    COLOR_PALETTE[8] = "#3d2617"  # dark umber
    COLOR_PALETTE[9] = "#d4b085"  # light terracotta
    COLOR_PALETTE[10] = "#eef2e6"  # pale cream
}
else if (arg_color_palette eq "moss_stone") {
    # Curated: forest moss greens over cool stone and charcoal grays.
    COLOR_PALETTE[1] = "#4a5d3a"  # moss green
    COLOR_PALETTE[2] = "#8c8577"  # stone gray
    COLOR_PALETTE[3] = "#6f8c5a"  # fresh moss
    COLOR_PALETTE[4] = "#5c5851"  # dark stone
    COLOR_PALETTE[5] = "#2c3820"  # deep moss
    COLOR_PALETTE[6] = "#b0b3a4"  # pale stone
    COLOR_PALETTE[7] = "#9bb37f"  # pale moss
    COLOR_PALETTE[8] = "#3a3833"  # charcoal stone
    COLOR_PALETTE[9] = "#7a9a5f"  # spring moss
    COLOR_PALETTE[10] = "#e2e4d9"  # pale mist
}
else if (arg_color_palette eq "cherry_blossom") {
    # Curated: blossom pink and berry over dark branch brown.
    COLOR_PALETTE[1] = "#c96c8a"  # cherry pink
    COLOR_PALETTE[2] = "#5c3a2e"  # branch brown
    COLOR_PALETTE[3] = "#e8a3bc"  # pale blossom
    COLOR_PALETTE[4] = "#3a2318"  # dark bark
    COLOR_PALETTE[5] = "#a13f5f"  # deep cherry
    COLOR_PALETTE[6] = "#f2cfdd"  # pale pink
    COLOR_PALETTE[7] = "#7a5540"  # bark tan
    COLOR_PALETTE[8] = "#8c2d4f"  # berry
    COLOR_PALETTE[9] = "#d99fb0"  # dusty pink
    COLOR_PALETTE[10] = "#f7e8ee"  # white blossom
}
else if (arg_color_palette eq "harbor_fog") {
    # Curated: harbor blue-grays and fog neutrals with a small pier-green accent.
    COLOR_PALETTE[1] = "#3d5a6b"  # harbor blue
    COLOR_PALETTE[2] = "#a8b0b5"  # fog gray
    COLOR_PALETTE[3] = "#5c7d8f"  # mid harbor blue
    COLOR_PALETTE[4] = "#233a45"  # deep harbor
    COLOR_PALETTE[5] = "#c9d0d3"  # pale fog
    COLOR_PALETTE[6] = "#7d98a3"  # steel blue fog
    COLOR_PALETTE[7] = "#0f1e26"  # night harbor
    COLOR_PALETTE[8] = "#8fa8b3"  # light steel
    COLOR_PALETTE[9] = "#e6ebec"  # white fog
    COLOR_PALETTE[10] = "#4a6b45"  # pier green accent
}
else if (arg_color_palette eq "ember_glow") {
    # Curated: smoldering charcoal reds through fire orange to glowing gold.
    COLOR_PALETTE[1] = "#7a1710"  # dark ember
    COLOR_PALETTE[2] = "#e8621f"  # ember orange
    COLOR_PALETTE[3] = "#4a0d08"  # charred red
    COLOR_PALETTE[4] = "#f2a33c"  # glow gold
    COLOR_PALETTE[5] = "#1a0503"  # ash black
    COLOR_PALETTE[6] = "#c9391a"  # fire red
    COLOR_PALETTE[7] = "#f7cf8a"  # pale glow
    COLOR_PALETTE[8] = "#8c2510"  # deep fire
    COLOR_PALETTE[9] = "#3a1408"  # smolder
    COLOR_PALETTE[10] = "#fde3b8"  # pale ember
}
else if (arg_color_palette eq "sage_clay") {
    # Curated: muted sage green with warm clay and terracotta accents.
    COLOR_PALETTE[1] = "#7a8c6a"  # sage green
    COLOR_PALETTE[2] = "#c9764a"  # clay orange
    COLOR_PALETTE[3] = "#a3b393"  # pale sage
    COLOR_PALETTE[4] = "#8c4a2c"  # deep clay
    COLOR_PALETTE[5] = "#4a5c3d"  # dark sage
    COLOR_PALETTE[6] = "#e8b48f"  # pale clay
    COLOR_PALETTE[7] = "#5c6b4a"  # mid sage
    COLOR_PALETTE[8] = "#6b3418"  # dark terracotta
    COLOR_PALETTE[9] = "#c4cfb5"  # pale sage mist
    COLOR_PALETTE[10] = "#f2e0c9"  # cream
}
else if (arg_color_palette eq "deep_sea") {
    # Curated: abyssal navy and black with bioluminescent teal accents.
    COLOR_PALETTE[1] = "#0a2f3d"  # deep sea blue
    COLOR_PALETTE[2] = "#12736b"  # abyss teal
    COLOR_PALETTE[3] = "#1a4a5c"  # mid sea blue
    COLOR_PALETTE[4] = "#2ba89a"  # bioluminescent teal
    COLOR_PALETTE[5] = "#04141c"  # abyss black
    COLOR_PALETTE[6] = "#5fbfae"  # pale teal glow
    COLOR_PALETTE[7] = "#0d3d4a"  # dark teal
    COLOR_PALETTE[8] = "#8fd9cb"  # pale bioluminescent
    COLOR_PALETTE[9] = "#020a0e"  # trench black
    COLOR_PALETTE[10] = "#c9ece5"  # pale sea mist
}
else if (arg_color_palette eq "grayscale") {
    # Monochrome output or print-friendly fallback.
    COLOR_PALETTE[1] = "#111111"  # very dark gray
    COLOR_PALETTE[2] = "#333333"  # dark gray
    COLOR_PALETTE[3] = "#555555"  # medium dark gray
    COLOR_PALETTE[4] = "#777777"  # medium gray
    COLOR_PALETTE[5] = "#999999"  # medium light gray
    COLOR_PALETTE[6] = "#aaaaaa"  # light gray
    COLOR_PALETTE[7] = "#bbbbbb"  # very light gray
    COLOR_PALETTE[8] = "#cccccc"  # extremely light gray
    COLOR_PALETTE[9] = "#dddddd"  # almost white gray
    COLOR_PALETTE[10] = "#eeeeee"  # white gray
}
else {
    # Unknown palette name: use the default rather than leaving the array empty.
    print sprintf("Unknown palette '%s'; using research_muted", arg_color_palette)
    COLOR_PALETTE[1] = "#817988"
    COLOR_PALETTE[2] = "#98adc9"
    COLOR_PALETTE[3] = "#595163"
    COLOR_PALETTE[4] = "#44667f"
    COLOR_PALETTE[5] = "#b2a5b3"
    COLOR_PALETTE[6] = "#9095b0"
    COLOR_PALETTE[7] = "#30303e"
    COLOR_PALETTE[8] = "#053348"
    COLOR_PALETTE[9] = "#34576b"
    COLOR_PALETTE[10] = "#ebe1e7"
}
