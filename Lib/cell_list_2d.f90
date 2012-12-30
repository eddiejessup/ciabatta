module cell_list_2d
    use utils
    implicit none

    integer, allocatable, private :: cl(:, :, :), cli(:, :)
    integer, allocatable :: inters(:, :), intersi(:)
contains

subroutine initialise_cl(n, m)
    integer, intent(in) :: n, m

    if (allocated(cl) .and. (size(cl, 1) /= n .or. size(cl, 2) /= m)) then
        deallocate(cl)
        deallocate(cli)
    end if
    if (.not. allocated(cl)) then
        allocate(cl(n, m, m))
        allocate(cli(size(cl, 2), size(cl, 3)))
    end if
end subroutine

subroutine initialise_inters(n)
    integer, intent(in) :: n

    if (allocated(inters) .and. size(inters, 1) /= n) then
        deallocate(inters)
        deallocate(intersi)
    end if
    if (.not. allocated(inters)) then
        allocate(inters(n, n))
        allocate(intersi(size(inters, 2)))
    end if
end subroutine

subroutine make_inters(r, l, r_cut)
    real, intent(in) :: r(:, :), l, r_cut
    integer :: inds(size(r, 1), size(r, 2)), m, x, y, x_inc, x_dec, y_inc, y_dec, i, i_cl
    real :: l_half, r_cut_sq

    m = int(floor(l / r_cut))
    l_half = l / 2.0
    r_cut_sq = r_cut ** 2
    call initialise_cl(size(r, 2), m)
    call initialise_inters(size(r, 2))

    inds = r_to_inds(r, l, m)

    cli = 0
    do i = 1, size(r, 2)
        cli(inds(1, i), inds(2, i)) = cli(inds(1, i), inds(2, i)) + 1
        cl(cli(inds(1, i), inds(2, i)), inds(1, i), inds(2, i)) = i
    end do

    intersi = 0
    do x = 1, m
        do y = 1, m
            x_inc = i_wrap(x + 1, m)
            x_dec = i_wrap(x - 1, m)
            y_inc = i_wrap(y + 1, m)
            y_dec = i_wrap(y - 1, m)
            do i_cl = 1, cli(x, y)
                i = cl(i_cl, x, y)
                call core(i, x, y)
                call core(i, x, y_inc)
                call core(i, x, y_dec)
                call core(i, x_inc, y)
                call core(i, x_dec, y)
                call core(i, x_inc, y_inc)
                call core(i, x_inc, y_dec)
                call core(i, x_dec, y_inc)
                call core(i, x_dec, y_dec)
            end do
        end do
    end do
contains

subroutine core(i, x, y)
    integer, intent(in) :: i, x, y
    integer :: i_target_cl, i_target
    do i_target_cl = 1, cli(x, y)
        i_target = cl(i_target_cl, x, y)
        if (i_target /= i .and. r_sep_sq(r(:, i), r(:, i_target), l, l_half) < r_cut_sq) then
            intersi(i) = intersi(i) + 1
            inters(intersi(i), i) = i_target
        end if
    end do
    return
end subroutine

end subroutine

subroutine make_inters_direct(r, l, r_cut)
    real, intent(in) :: r(:, :), l, r_cut
    integer :: i, i_target, m
    real :: l_half, r_cut_sq

    m = int(floor(l / r_cut))
    l_half = l / 2.0
    r_cut_sq = r_cut ** 2
    call initialise_inters(size(r, 2))

    intersi = 0
    do i = 1, size(r, 2)
        do i_target = i + 1, size(r, 2)
            if (r_sep_sq(r(:, i), r(:, i_target), l, l_half) < r_cut_sq) then
                intersi(i) = intersi(i) + 1
                inters(intersi(i), i) = i_target
                intersi(i_target) = intersi(i_target) + 1
                inters(intersi(i_target), i_target) = i
            end if
        end do
    end do
end subroutine

end module
