package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.Reservation;

import java.util.List;

public interface ReservationService extends IService<Reservation> {

    String createReservation(Reservation reservation);

    void cancelReservation(Long reservationId, Long customerId, String reason);

    List<Reservation> listByCustomerId(Long customerId);

    void confirm(Long reservationId);

    void reject(Long reservationId);

    void arrive(Long reservationId);

    void complete(Long reservationId);

    void noShow(Long reservationId);
}
