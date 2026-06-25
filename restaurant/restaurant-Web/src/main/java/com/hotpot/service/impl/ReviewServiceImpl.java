package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.common.BusinessException;
import com.hotpot.entity.Reservation;
import com.hotpot.entity.Review;
import com.hotpot.mapper.ReservationMapper;
import com.hotpot.mapper.ReviewMapper;
import com.hotpot.service.ReviewService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ReviewServiceImpl extends ServiceImpl<ReviewMapper, Review> implements ReviewService {

    private final ReservationMapper reservationMapper;

    @Override
    public void createReview(Review review) {
        Reservation reservation = reservationMapper.selectById(review.getReservationId());
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        if (!reservation.getCustomerId().equals(review.getCustomerId())) {
            throw new BusinessException("无权评价该预订");
        }
        if (reservation.getStatus() != 3) {
            throw new BusinessException("仅已完成状态的预订可评价");
        }
        review.setIsVisible(1);
        save(review);
    }

    @Override
    public List<Review> listByStoreId(Long storeId, int pageNum, int pageSize) {
        Page<Review> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<Review> wrapper = new LambdaQueryWrapper<Review>()
                .eq(Review::getStoreId, storeId)
                .eq(Review::getIsVisible, 1)
                .orderByDesc(Review::getCreateTime);
        return page(page, wrapper).getRecords();
    }
}
