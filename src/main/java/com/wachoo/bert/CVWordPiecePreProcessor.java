package com.wachoo.bert;

import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.BertWordPiecePreProcessor;

import java.text.Normalizer;
import java.util.Map;

/**
 * @Author : wachoo
 * @Time : 2021/4/1 16:18
 */
public class CVWordPiecePreProcessor extends BertWordPiecePreProcessor {

    public CVWordPiecePreProcessor(boolean lowerCase, boolean stripAccents, Map<String, Integer> vocab) {
        super(lowerCase, stripAccents, vocab);
    }


    @Override
    public String preProcess(String token) {
        if (stripAccents) {
            token = Normalizer.normalize(token, Normalizer.Form.NFD);
        }

        int n = token.codePointCount(0, token.length());
        StringBuilder sb = new StringBuilder();
        int charOffset = 0;
        int cps = 0;
        while (cps++ < n) {
            int cp = token.codePointAt(charOffset);
            charOffset += Character.charCount(cp);

            //Remove control characters and accents
            if (cp == 0 || cp == REPLACEMENT_CHAR || isControlCharacter(cp) || (stripAccents && Character.getType(cp) == Character.NON_SPACING_MARK)) {
                continue;
            }

            //Convert to lower case if necessary
            if (lowerCase) {
                cp = Character.toLowerCase(cp);
            }

            //Replace whitespace chars with space
            if (isWhiteSpace(cp)) {
                sb.append(' ');
                continue;
            }

            if (charSet != null && !charSet.contains(cp)) {
                //Skip unknown character (out-of-vocab - though this should rarely happen)
                continue;
            }

            //Handle Chinese and other characters
            if (isChineseCharacter(cp)) {
//                sb.append(' ');
                sb.appendCodePoint(cp);
//                sb.append(' ');
                continue;
            }

            //All other characters - keep
            sb.appendCodePoint(cp);
        }

        return sb.toString();
    }
}
