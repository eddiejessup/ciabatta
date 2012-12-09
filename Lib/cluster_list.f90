subroutine cluster_list(r_raw, R_cut_raw, L, list, N, D)
    implicit none

    integer, intent(in) :: N, D
    real, intent(in) :: r_raw(N, D), R_cut_raw, L
    integer, intent(out) :: list(N)

    real :: r(N, D), R_cut_sq, r_j(D), r_j_k(D)
    integer :: i, j, k, list_k

    ! scale distances to internal units
    r = r_raw / L
    R_cut_sq = (R_cut_raw / L) ** 2

    do i = 1, N
        list(i) = i
    end do

    do i = 1, N - 1
        if (i == list(i)) then
            j = i
            r_j = r(j, :)
            do k = i + 1, N
                list_k = list(k)
                if (list_k == k) then
                    r_j_k = r_j - r(k, :)
                    r_j_k = r_j_k - anint(r_j_k)
                    if (sum(r_j_k ** 2) <= r_cut_sq) then
                        list(k) = list(j)
                        list(j) = list_k
                    end if
                end if
            end do

            j = list(j)
            r_j = r(j, :)

            do while (j /= i)
                do k = i + 1, N
                    list_k = list(k)
                    if (list_k == k) then
                        r_j_k = r_j - r(k, :)
                        r_j_k = r_j_k - anint(r_j_k)
                        if (sum(r_j_k ** 2) < r_cut_sq) then
                            list(k) = list(j)
                            list(j) = list_k
                        end if
                    end if
                end do
                j = list(j)
                r_j = r(j, :)
            end do
        end if
    end do
end subroutine cluster_list
