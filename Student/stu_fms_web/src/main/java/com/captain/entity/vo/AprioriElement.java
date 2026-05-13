package com.captain.entity.vo;

import lombok.Data;

import java.util.Objects;

/**
 * Apriori分析里面的元素
 */
@Data
public class AprioriElement {
    private Long id;

    private String value;

    /**
     * 重写equal方法，直接判断值是否相同
     * @param object
     * @return
     */
    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        AprioriElement that = (AprioriElement) object;
        return value.equals(that.value);
    }

    @Override
    public int hashCode() {
        return Objects.hash(value);
    }
}
