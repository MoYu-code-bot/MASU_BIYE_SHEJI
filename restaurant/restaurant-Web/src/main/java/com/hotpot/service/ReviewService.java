package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.Review;

import java.util.List;

public interface ReviewService extends IService<Review> {

    void createReview(Review review);

    List<Review> listByStoreId(Long storeId, int pageNum, int pageSize);
}
