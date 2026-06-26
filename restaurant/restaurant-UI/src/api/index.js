import request from '@/utils/request'

// ==================== Auth ====================
export const customerLogin = (data) => request.post('/api/auth/login', data)
export const customerRegister = (data) => request.post('/api/auth/register', data)
export const adminLogin = (data) => request.post('/admin/auth/login', data)

// ==================== Store (Customer) ====================
export const getStores = () => request.get('/api/stores/list')
export const getStoreDetail = (storeId) => request.get('/api/stores/detail', { params: { storeId } })
export const getStoreDishes = (storeId) => request.get('/api/stores/dishes', { params: { storeId } })
export const getStoreSlots = (storeId) => request.get('/api/stores/slots', { params: { storeId } })
export const getStoreReviews = (storeId, params) => request.get('/api/stores/reviews', { params: { storeId, ...params } })

// ==================== Category (Customer) ====================
export const getCategories = () => request.get('/api/categories/list')

// ==================== Banner (Customer) ====================
export const getBanners = () => request.get('/api/banners/list')

// ==================== Announcement (Customer) ====================
export const getAnnouncements = (storeId) => {
  const params = storeId ? { storeId } : {}
  return request.get('/api/announcements/list', { params })
}

// ==================== Reservation (Customer) ====================
export const createReservation = (data) => request.post('/api/reservations/create', data)
export const getMyReservations = () => request.get('/api/reservations/list')
export const getReservationDetail = (reservationId) => request.get('/api/reservations/detail', { params: { reservationId } })
export const cancelReservation = (reservationId, reason) => request.put('/api/reservations/cancel', null, { params: { reservationId, reason } })

// ==================== Review (Customer) ====================
export const createReview = (data) => request.post('/api/reviews/create', data)

// ==================== Admin Store ====================
export const adminGetStores = (params) => request.get('/admin/stores/list', { params })
export const adminAddStore = (data) => request.post('/admin/stores/create', data)
export const adminUpdateStore = (storeId, data) => request.put('/admin/stores/update', data, { params: { storeId } })
export const adminDeleteStore = (storeId) => request.delete('/admin/stores/delete', { params: { storeId } })

// Aliases used by pages
export const createStore = adminAddStore
export const updateStore = (store) => request.put('/admin/stores/update', store, { params: { storeId: store.id } })
export const deleteStore = adminDeleteStore

// ==================== Admin Category ====================
export const adminGetCategories = () => request.get('/admin/categories/list')
export const adminAddCategory = (data) => request.post('/admin/categories/create', data)
export const adminUpdateCategory = (categoryId, data) => request.put('/admin/categories/update', data, { params: { categoryId } })
export const adminDeleteCategory = (categoryId) => request.delete('/admin/categories/delete', { params: { categoryId } })

// Aliases
export const createCategory = adminAddCategory
export const updateCategory = (category) => request.put('/admin/categories/update', category, { params: { categoryId: category.id } })
export const deleteCategory = adminDeleteCategory

// ==================== Admin Dish ====================
export const adminGetDishes = (params) => request.get('/admin/dishes/list', { params })
export const adminAddDish = (data) => request.post('/admin/dishes/create', data)
export const adminUpdateDish = (dishId, data) => request.put('/admin/dishes/update', data, { params: { dishId } })
export const adminUpdateDishStatus = (dishId, status) => request.put('/admin/dishes/updateStatus', null, { params: { dishId, status } })

// Aliases used by DishManage page
export const getDishes = adminGetDishes
export const createDish = adminAddDish
export const updateDish = (dish) => request.put('/admin/dishes/update', dish, { params: { dishId: dish.id } })
export const deleteDish = (dishId) => request.delete('/admin/dishes/delete', { params: { dishId } })

// ==================== Admin Reservation ====================
export const adminGetReservations = (params) => request.get('/admin/reservations/list', { params })
export const adminConfirmReservation = (reservationId) => request.put('/admin/reservations/confirm', null, { params: { reservationId } })
export const adminRejectReservation = (reservationId) => request.put('/admin/reservations/reject', null, { params: { reservationId } })
export const adminArriveReservation = (reservationId) => request.put('/admin/reservations/arrive', null, { params: { reservationId } })
export const adminCompleteReservation = (reservationId) => request.put('/admin/reservations/complete', null, { params: { reservationId } })

// Aliases used by ReservationManage page
export const getReservations = adminGetReservations
export const updateReservationStatus = (reservationId, action) => {
  const map = {
    confirm: adminConfirmReservation,
    reject: adminRejectReservation,
    arrive: adminArriveReservation,
    complete: adminCompleteReservation,
    noshow: (id) => request.put(`/admin/reservations/noshow?reservationId=${id}`)
  }
  return map[action](reservationId)
}

// ==================== Admin Review ====================
export const adminGetReviews = (params) => request.get('/admin/reviews/list', { params })
export const adminUpdateReviewVisible = (reviewId, isVisible) => request.put('/admin/reviews/updateVisible', null, { params: { reviewId, isVisible } })

// Aliases used by ReviewManage page
export const getReviews = adminGetReviews
export const updateReview = (review) => request.put('/admin/reviews/updateVisible', null, { params: { reviewId: review.id, isVisible: review.isVisible } })

// ==================== Admin Banner ====================
export const adminGetBanners = () => request.get('/admin/banners/list')
export const adminAddBanner = (data) => request.post('/admin/banners/create', data)
export const adminUpdateBanner = (bannerId, data) => request.put('/admin/banners/update', data, { params: { bannerId } })
export const adminDeleteBanner = (bannerId) => request.delete('/admin/banners/delete', { params: { bannerId } })

// Aliases
export const createBanner = adminAddBanner
export const updateBanner = (banner) => request.put('/admin/banners/update', banner, { params: { bannerId: banner.id } })
export const deleteBanner = adminDeleteBanner

// ==================== Admin Announcement ====================
export const adminGetAnnouncements = (storeId) => request.get('/admin/announcements/list', { params: { storeId } })
export const adminAddAnnouncement = (data) => request.post('/admin/announcements/create', data)
export const adminUpdateAnnouncement = (announcementId, data) => request.put('/admin/announcements/update', data, { params: { announcementId } })
export const adminDeleteAnnouncement = (announcementId) => request.delete('/admin/announcements/delete', { params: { announcementId } })

// Aliases
export const createAnnouncement = adminAddAnnouncement
export const updateAnnouncement = (ann) => request.put('/admin/announcements/update', ann, { params: { announcementId: ann.id } })
export const deleteAnnouncement = adminDeleteAnnouncement

// ==================== Admin Dashboard ====================
export const getDashboard = () => request.get('/admin/dashboard/overview')

// ==================== Admin User Management ====================
export const adminGetUsers = (params) => request.get('/admin/users/list', { params })
export const adminUpdateUserStatus = (userId, status) => request.put('/admin/users/updateStatus', null, { params: { userId, status } })
export const adminCreateUser = (user) => request.post('/admin/users/create', user)
export const adminUpdateUser = (userId, user) => request.put(`/admin/users/${userId}`, user)
export const adminDeleteUser = (userId) => request.delete(`/admin/users/${userId}`)

// Aliases used by SysUserManage page
export const getSysUsers = adminGetUsers
export const createSysUser = adminCreateUser
export const updateSysUserInfo = adminUpdateUser
export const deleteSysUser = adminDeleteUser
export const updateSysUser = (user) => request.put('/admin/users/updateStatus', null, { params: { userId: user.id, status: user.status } })

// ==================== Time Slots ====================
export const getTimeSlots = (storeId) => request.get('/api/stores/slots', { params: { storeId } })
