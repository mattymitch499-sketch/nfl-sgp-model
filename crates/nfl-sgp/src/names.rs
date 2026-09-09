//! One spelling per player, so quotes from different books group as one leg.
//!
//! Books disagree on accents and punctuation ("Ja'Marr Chase" vs "JaMarr
//! Chase", "A.J. Brown" vs "AJ Brown"), and nflverse abbreviates first names
//! in `player_name` while spelling them out in `player_display_name`. The
//! folded key is lowercase ASCII letters and digits with single spaces; every
//! join in the project goes through it. Lifted from the baseball crate, where
//! the failure it prevents was two single-book anchors for one pitcher that
//! both failed the two-book consensus bar.

/// Lowercase, accent-folded, punctuation collapsed to single spaces.
pub fn normalize_name(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len());
    for character in name.chars().flat_map(char::to_lowercase) {
        let folded = match character {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other,
        };
        if folded.is_ascii_alphanumeric() {
            normalized.push(folded);
        } else if !normalized.ends_with(' ') && !normalized.is_empty() {
            normalized.push(' ');
        }
    }
    normalized.trim().to_string()
}

/// The **join key**: [`normalize_name`] with the separators removed entirely,
/// so `"Ja'Marr Chase"`, `"JaMarr Chase"` and `"Ja Marr Chase"` are one player.
///
/// This exists because [`normalize_name`] alone is not enough, and the way it
/// fails is silent. It turns punctuation into a space, which folds accents
/// correctly (`"Sánchez"` and `"Sanchez"` agree) but leaves apostrophes and
/// periods splitting a name differently depending on how a book typed it:
/// `"Ja'Marr"` becomes `ja marr` while `"JaMarr"` becomes `jamarr`, and
/// `"A.J. Brown"` becomes `a j brown` while `"AJ Brown"` becomes `aj brown`.
/// Two books spelling one receiver two ways would then produce two
/// single-book consensus rows, each failing the two-book bar, and the leg would
/// come back unscoreable for no visible reason.
///
/// Dropping every separator collapses all of those onto one key. The theoretical
/// cost is a collision between two players whose letters coincide once spaces
/// go (`"Mike Evans"` vs a hypothetical `"Mi Keevans"`), which does not happen
/// in a league of 1,700 names; the real cost of the alternative is rows quietly
/// dropping out of the study.
///
/// Use this for every join between the sheet, the leg board and the consensus.
/// Use [`normalize_name`] when a human has to read the result.
pub fn join_key(name: &str) -> String {
    normalize_name(name).replace(' ', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apostrophes_and_periods_fold_to_one_key() {
        assert_eq!(normalize_name("Ja'Marr Chase"), "ja marr chase");
        assert_eq!(normalize_name("A.J. Brown"), "a j brown");
        assert_eq!(normalize_name("AJ Brown"), "aj brown");
    }

    #[test]
    fn accents_fold_and_whitespace_collapses() {
        assert_eq!(normalize_name("  José   Ramírez "), "jose ramirez");
        assert_eq!(normalize_name("Cristopher Sánchez"), "cristopher sanchez");
    }

    #[test]
    fn empty_and_punctuation_only_are_empty() {
        assert_eq!(normalize_name(""), "");
        assert_eq!(normalize_name("..."), "");
        assert_eq!(join_key(""), "");
        assert_eq!(join_key("..."), "");
    }

    /// The spellings that actually differ between books, all landing on one
    /// key. Each pair here is a leg that would otherwise have come back
    /// unscoreable because neither spelling reached two books on its own.
    #[test]
    fn the_join_key_survives_every_spelling_books_disagree_on() {
        for (left, right) in [
            ("Ja'Marr Chase", "JaMarr Chase"),
            ("A.J. Brown", "AJ Brown"),
            ("Amon-Ra St. Brown", "Amon Ra St Brown"),
            ("D'Andre Swift", "DAndre Swift"),
            ("Cristopher Sánchez", "Cristopher Sanchez"),
            ("  Patrick   Mahomes ", "Patrick Mahomes"),
        ] {
            assert_eq!(
                join_key(left),
                join_key(right),
                "{left:?} and {right:?} must be one player"
            );
        }
        assert_eq!(join_key("Ja'Marr Chase"), "jamarrchase");
    }

    /// It must still tell genuinely different players apart.
    #[test]
    fn the_join_key_does_not_merge_different_players() {
        assert_ne!(join_key("Josh Allen"), join_key("Keenan Allen"));
        assert_ne!(join_key("Mike Williams"), join_key("Mike Evans"));
        assert_ne!(join_key("Kansas City Chiefs"), join_key("Kansas City"));
    }
}
